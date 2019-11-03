import random
import numpy as np
import neat.utils as utils
from neat.genotype.genome import Genome
from neat.species import Species
from neat.crossover import crossover
from neat.mutation import mutate
from neat.visualize import draw_net

from tqdm import tqdm
import sys
import itertools as it

import copy


def pool_func(arg):
	fitness_func, genome, shared_data, gi, pi = arg
	fitness = max(0, fitness_func(genome, shared_data))
	print("Generation: %d, Genome: %d --> Fitness: %f" % (gi, pi, fitness))
	sys.stdout.flush()
	draw_net(genome, view=False, \
		filename="./images/solution-g%d-p%d"%(gi, pi), show_disabled=True)
	return fitness
	
class Population:
	__global_innovation_number = 0
	current_gen_innovation = []  # Can be reset after each generation according to paper

	def __init__(self, config):
		self.Config = config()
		self.population = self.set_initial_population()
		self.species = []

		for genome in self.population:
			self.speciate(genome, 0)

	def run(self, pool=None, shared_data=None):
		allgenfitnesses = []
		for generation in range(1, self.Config.NUMBER_OF_GENERATIONS):
			# ****** BYME: Neuro-evolution accures here *******
			# Get Fitness of Every Genome
			if pool != None:
				ll = len(self.population)
				args = zip(list(it.repeat(self.Config.fitness_fn, ll)), \
					self.population, list(it.repeat(shared_data, ll)), \
					list(it.repeat(generation, ll)), range(ll))
				fitnesses = list(pool.map(pool_func, args))
				for genome, fitness in zip(self.population, fitnesses):
					genome.fitness = fitness				
			else:
				for genome in tqdm(self.population):
					genome.fitness = max(0, self.Config.fitness_fn(genome))
				
			allfitnesses_onegen = [g.fitness for g in self.population]
			allfitnesses_onegen.sort()
			allgenfitnesses.append(allfitnesses_onegen)

				
			best_genome = utils.get_best_genome(self.population)
			draw_net(best_genome, view=False, \
				filename="./images/solution-best-g%d"%(generation), show_disabled=True)
				
			# Reproduce
			all_fitnesses = []
			remaining_species = []

			for species, is_stagnant in Species.stagnation(self.species, generation):
				if is_stagnant:
					self.species.remove(species)
				else:
					all_fitnesses.extend(g.fitness for g in species.members)
					remaining_species.append(species)

			min_fitness = min(all_fitnesses)
			max_fitness = max(all_fitnesses)

			fit_range = max(1.0, (max_fitness-min_fitness))
			for species in remaining_species:
				# Set adjusted fitness
				avg_species_fitness = np.mean([g.fitness for g in species.members])
				species.adjusted_fitness = (avg_species_fitness - min_fitness) / fit_range

			adj_fitnesses = [s.adjusted_fitness for s in remaining_species]
			adj_fitness_sum = sum(adj_fitnesses)

			# Get the number of offspring for each species
			new_population = []
			for species in remaining_species:
				if species.adjusted_fitness > 0:
					size = max(2, int((species.adjusted_fitness/adj_fitness_sum) * self.Config.POPULATION_SIZE))
				else:
					size = 2

				# sort current members in order of descending fitness
				cur_members = species.members
				cur_members.sort(key=lambda g: g.fitness, reverse=True)
				species.members = []  # reset

				# save top individual in species
				new_population.append(cur_members[0])
				size -= 1

				# Only allow top x% to reproduce
				purge_index = int(self.Config.PERCENTAGE_TO_SAVE * len(cur_members))
				purge_index = max(2, purge_index)
				cur_members = cur_members[:purge_index]

				for i in range(size):
					parent_1 = random.choice(cur_members)
					parent_2 = random.choice(cur_members)

					child = crossover(parent_1, parent_2, self.Config)
					mutate(child, self.Config)
					new_population.append(child)

			# Set new population
			self.population = new_population
			Population.current_gen_innovation = []

			# Speciate
			for genome in self.population:
				self.speciate(genome, generation)

			if best_genome.fitness >= self.Config.FITNESS_THRESHOLD:
				o = open("allgenfitnesses.txt", "w")
				for allfitnesses_onegen in allgenfitnesses:
					o.write(str(allfitnesses_onegen) + "\n")
				o.close()
				return best_genome, generation

			# Generation Stats
			if self.Config.VERBOSE:
				print('Finished Generation',  generation)
				print('Best Genome Fitness:', best_genome.fitness)
				if hasattr(best_genome, "avgloss"):
					print('Best Genome Loss:', best_genome.avgloss)
				print('Best Genome Length',   len(best_genome.connection_genes))
				print()

		
		o = open("allgenfitnesses.txt", "w")		
		for allfitnesses_onegen in allgenfitnesses:
			o.write(str(allfitnesses_onegen) + "\n")
		o.close()
			
		return None, None

	def speciate(self, genome, generation):
		"""
		Places Genome into proper species - index
		:param genome: Genome be speciated
		:param generation: Number of generation this speciation is occuring at
		:return: None
		"""
		for species in self.species:
			if Species.species_distance(genome, species.model_genome) <= self.Config.SPECIATION_THRESHOLD:
				genome.species = species.id
				species.members.append(genome)
				return

		# Did not match any current species. Create a new one
		new_species = Species(len(self.species), genome, generation)
		genome.species = new_species.id
		new_species.members.append(genome)
		self.species.append(new_species)

	def assign_new_model_genomes(self, species):
		species_pop = self.get_genomes_in_species(species.id)
		species.model_genome = random.choice(species_pop)

	def get_genomes_in_species(self, species_id):
		return [g for g in self.population if g.species == species_id]

	def set_initial_population(self):
		pop = []
		for i in range(self.Config.POPULATION_SIZE):
			new_genome = Genome()
			inputs = []
			outputs = []
			bias = None

			# Create nodes
			for j in range(self.Config.NUM_INPUTS):
				n = new_genome.add_node_gene('input')
				inputs.append(n)

			for j in range(self.Config.NUM_OUTPUTS):
				n = new_genome.add_node_gene('output')
				outputs.append(n)

			if self.Config.USE_BIAS:
				bias = new_genome.add_node_gene('bias')

			# Create connections
			for input in inputs:
				for output in outputs:
					new_genome.add_connection_gene(input.id, output.id)

			if bias is not None:
				for output in outputs:
					new_genome.add_connection_gene(bias.id, output.id)

			pop.append(new_genome)

		return pop

	@staticmethod
	def get_new_innovation_num():
		# Ensures that innovation numbers are being counted correctly
		# This should be the only way to get a new innovation numbers
		ret = Population.__global_innovation_number
		Population.__global_innovation_number += 1
		return ret
