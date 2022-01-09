import json
import time
import random

import numpy as np
import pandas as pd
from enum import Enum
from typing import List, Tuple


class Response(Enum):
	"""
	The possible responses to guessing a character in a possition. 
		- MATCH means the right character in the right position. (green)
		- MISS means the character is in the word, but the position is wrong (yellow)
		- WRONG means the character is not in the word at all (red)
	"""
	MATCH = 1
	MISS = 2
	WRONG = 3


WORD_LENGTH = 5


def calc_char_dist(curr_pool: List[str]) -> pd.DataFrame:
	"""
	Calculate the distribution P[X, i] = probability that character X appears
	at position i. The final table will be 5x26
	"""
	# explode the list of words into a list of list of chars
	pool_char_list = pd.DataFrame([list(w) for w in curr_pool])
	
	# define a function to get the frequency of each character
	get_frequency = lambda x: x.value_counts(normalize=True)

	# by default, pandas puts NaN when the count is 0, but we want 0
	return pool_char_list.apply(get_frequency).fillna(0)

def calc_exp_info_table(curr_pool: List[str]) -> pd.DataFrame:
	"""
	Calculate the table ExpInfo[X, i] = expected number of possible solutions eliminated
	if for guessing character X in position i.

	To get the expected info, first calculate the probability that guessing
	character X at position i will result in response R, then multiply by the
	number of words eliminated by that response.
	"""

	char_dist = calc_char_dist(curr_pool)
	# extract the raw data from the dataframe, I think this will make things
	# faster
	char_dist_arr = char_dist.values
	# precompute the probability table P[X] = probability that character X occurs
	# at all
	total_char_dist = np.tile(char_dist_arr.sum(axis=1), (WORD_LENGTH, 1)).T

	# a MATCH will eliminate all words that don't have character X in position i
	# this is equivalent to the probability that character in position i is not
	# X, times the number of words
	match_info = (1 - char_dist_arr) * len(curr_pool)
	# the probaility of getting a match is the same as char_dist
	match_exp_info = char_dist_arr * match_info

	# a MISS will eliminate all the words that don't have character X, and all
	# the words that have character X in position i
	# these conditions never happen together, so there is no risk of double-counting
	# ie counting a word that satisfies both conditions as 2 words being removed
	miss_info = (char_dist_arr + (1 - total_char_dist)) * len(curr_pool)
	# the probaility of getting a MISS is the probability that char X is in the
	# word at all, minus the probability we get a MATCH
	miss_prob = total_char_dist - char_dist_arr
	miss_exp_info = miss_prob * miss_info

	# a WRONG eliminates all words that have the character X in them at all
	wrong_info = total_char_dist * len(curr_pool)
	wrong_prob = 1 - total_char_dist
	wrong_exp_info = wrong_prob * wrong_info


	# convert the raw array back to a table
	exp_info_arr = match_exp_info + miss_exp_info + wrong_exp_info
	exp_info_table = pd.DataFrame(
		exp_info_arr, columns=char_dist.columns, index=char_dist.index)
	return exp_info_table


def fast_calc_exp_info_table(curr_pool: List[str]) -> pd.DataFrame:
	"""
	Does the same as above, in a more mathematical way (maybe not actually
	that much faster). This formula was computed by writing out the full expression
	created by `calc_exp_info_table` and cancelling terms.
	"""
	char_dist = calc_char_dist(curr_pool)
	# extract the raw data from the dataframe, I think this will make things
	# faster
	char_dist_arr = char_dist.values
	# precompute the probability table P[X, i] = probability that character X occurs
	# at all
	total_char_dist = np.tile(char_dist_arr.sum(axis=1), (WORD_LENGTH, 1)).T

	# mathemagics
	exp_info_arr = char_dist_arr - char_dist_arr*(total_char_dist-char_dist_arr)

	# convert to a table
	exp_info_table = pd.DataFrame(
		exp_info_arr, columns=char_dist.columns, index=char_dist.index)
	return exp_info_table


def calc_score(curr_pool: List[str]) -> List[float]:
	"""
	Calculates the expected number of possible solutions that guessing each certain
	word will eliminate.
	"""
	exp_info_table = fast_calc_exp_info_table(curr_pool)

	score_list = []
	for word in curr_pool:
		score = sum(
			[exp_info_table.at[char, idx] for (idx,char) in enumerate(list(word))])
		score_list.append(score)

	return score_list

def get_best_word(curr_pool: List[str]) -> str:
	"""
	Gets the word with the best score
	"""
	score = calc_score(curr_pool)
	return curr_pool[np.argmax(score)]


def emit_response(answer: str, guess: str) -> List[Response]:
	"""
	Gives the response for each letter guessed.
	"""
	reponses = []
	for i in range(len(answer)):
		# correct charatcer and position
		if guess[i] == answer[i]:
			reponses.append(Response.MATCH)

		# character appears in the word, but not in that position
		elif guess[i] in answer:
			reponses.append(Response.MISS)

		# character is not the the word
		else:
			reponses.append(Response.WRONG)

	return reponses


def filter_from_response(guess: str, response_list: List[Response], curr_pool: List[str]) -> List[str]:
	"""
	Modifies the list `curr_pool` to remove the words that are no longer valid
	possible solutions
	"""
	for idx, (character, response) in enumerate(zip(guess, response_list)):
		if response == Response.MATCH:
			# if we get a MATCH for character X in position i, only words with 
			# w[i]=X can remain
			condition = lambda word: word[idx] == character

		elif response == Response.MISS:
			# if we get a MISS for character X in position i, then character X 
			# is in word, but not in position i, otherwise this would be a MATCH
			condition = lambda word: character in word and word[idx] != character

		else:
			# if we get a WRONG, then the character is simply not in the word
			condition = lambda word: character not in word

		curr_pool = list(filter(condition, curr_pool))

	return curr_pool


def solve(
	answer: str, 
	curr_pool: List[str], 
	start_word: str = None, 
	verbose: bool = False
) -> Tuple[float, int]:
	"""
	Solved the wordle and displays some info along the way. Return the time it
	took to filter/score and the number of guesses it took to get correct
	"""
	curr_guess = start_word
	# use the predetermined best if no start is given
	if curr_guess is None:
		curr_guess = "sores"

	if verbose:
		print(f"The answer is {answer}, let's go!")

	turn_num = 1
	calculation_time = 0		
	while curr_guess != answer:
		if verbose:
			print(f"Turn number: {turn_num}, guess: {curr_guess}, pool size: {len(curr_pool)}, calculation_time: {calculation_time}s")

		response_list = emit_response(answer, curr_guess)
		t0 = time.time()
		curr_pool = filter_from_response(curr_guess, response_list, curr_pool)
		curr_guess = get_best_word(curr_pool)
		calculation_time += (time.time() - t0)
		turn_num += 1

	if verbose:
		print(f"Woah, we guessed: {curr_guess}, in {turn_num} turns")

	return calculation_time, turn_num

def simulate(
	dictionary: List[str],
	num_simulations : int,
	start_word : str = None
) -> Tuple[np.ndarray, np.ndarray]:
	"""
	Simulates `num_simulations` wordles with different solutions. Return the 
	times and number of guesses for each simulation.
	"""
	num_guesses_list = []
	time_list = []

	for _ in range(num_simulations):
		answer = random.choice(dictionary)
		calc_time, num_guesses = solve(answer, dictionary, start_word=start_word)
		num_guesses_list.append(num_guesses)
		time_list.append(calc_time)

	return np.array(time_list), np.array(num_guesses_list)


def play_interactive(dictionary: List[str], max_turns: int):
	"""
	Plays the game based on your response, and gives you a recommended guess as 
	you go. You can use this to play the online version.
	"""
	curr_pool = dictionary
	for turn in range(max_turns):
		best_guess = get_best_word(curr_pool)
		print(f"Turn #{turn + 1}, my recommended guess is '{best_guess}'")
		print("Please input your guess")
		player_guess = input().strip()
		print(f"You guessed: {player_guess}")
		print("Now input the response, 1=green, 2=yellow, 3=grey/red, space seperated")
		player_response = input().strip().split()
		player_response = list(map(lambda x: Response(int(x)), player_response))

		if not (Response.MISS in player_response or Response.WRONG in player_response):
			print("Congrats")
			break

		else:
			curr_pool = filter_from_response(player_guess, player_response, curr_pool)



with open("dictionary.json") as json_file:
	dictionary = json.load(json_file)


all_times, all_num_guesses = simulate(dictionary, 100, "sores")


print("Computation time:")
print("\taverage: {:2f}".format(np.mean(all_times)))
print("\tmax: {}".format(np.max(all_times)))

print("Number of guesses:")
print("\taverage: {:2f}".format(np.mean(all_num_guesses)))
print("\tmax: {}".format(np.max(all_num_guesses)))