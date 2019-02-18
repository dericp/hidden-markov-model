#!/usr/bin/env python3

import argparse
import json
import math
from collections import Counter
from collections import defaultdict

import pandas as pd

import config as cfg
import calculate_accuracies


def main(args):
    if args.tune:
        # lists for CSV
        csv_K_vals = []
        csv_lambda_1s = []
        csv_lambda_2s = []
        csv_train_accs = []
        csv_dev_accs = []

        K_vals = [10**exp for exp in range(-5, 3)]
        for K in K_vals:
            print('tuning bigram HMM . . .')
            print()
            cfg.UNK_ADD_K = K
            label_to_words_to_counts, label_denoms, label_unigrams, label_bigrams = \
                train_hmm(cfg.train_path)
            print()
            for lambda1 in cfg.LAMBDA_1s:
                csv_K_vals.append(cfg.UNK_ADD_K)
                lambda2 = 1 - lambda1
                cfg.LAMBDA_1 = lambda1
                csv_lambda_1s.append(cfg.LAMBDA_1)
                cfg.LAMBDA_2 = lambda2
                csv_lambda_2s.append(cfg.LAMBDA_2)
                predict(cfg.train_path, cfg.train_out, label_to_words_to_counts,
                        label_denoms, label_unigrams, label_bigrams)
                train_acc = calculate_accuracies.evaluate(cfg.train_out, cfg.train_path)
                csv_train_accs.append(train_acc)
                predict(cfg.dev_path, cfg.dev_out, label_to_words_to_counts,
                        label_denoms, label_unigrams, label_bigrams)
                dev_acc = calculate_accuracies.evaluate(cfg.dev_out, cfg.dev_path)
                csv_dev_accs.append(dev_acc)
                print()
                if args.test:
                    predict(cfg.test_path, cfg.test_out, label_to_words_to_counts,
                            label_denoms, label_unigrams, label_bigrams)

        pd.DataFrame({'unk add-K': csv_K_vals,
                      'lambda 1': csv_lambda_1s,
                      'lambda 2': csv_lambda_2s,
                      'train accuracy': csv_train_accs,
                      'dev accuracy': csv_dev_accs}).to_csv('tuning-hmm.csv',
                                                            index=False)
    else:
        label_to_words_to_counts, label_denoms, label_unigrams, label_bigrams = \
                train_hmm(cfg.train_path)
        predict(cfg.train_path, cfg.train_out, label_to_words_to_counts,
                label_denoms, label_unigrams, label_bigrams)
        calculate_accuracies.evaluate(cfg.train_out, cfg.train_path)
        predict(cfg.dev_path, cfg.dev_out, label_to_words_to_counts,
                label_denoms, label_unigrams, label_bigrams)
        calculate_accuracies.evaluate(cfg.dev_out, cfg.dev_path)
        if args.test:
            predict(cfg.test_path, cfg.test_out, label_to_words_to_counts,
                    label_denoms, label_unigrams, label_bigrams)
            calculate_accuracies.evaluate(cfg.test_out, cfg.test_path)


def predict(input_path, output_path, label_to_words_to_counts, label_denoms,
        label_unigrams, label_bigrams):
    print('tagging', input_path, 'and saving to', output_path, '. . .')
    print('LAMBDA_1:', cfg.LAMBDA_1, 'LAMBDA_2:', cfg.LAMBDA_2)
    results = open(output_path, 'w')
    with open(input_path) as f:
        for line in f:
            tuples = json.loads(line)
            sequence = []
            labels = []
            for word, label in tuples:
                sequence.append(word)
                labels.append(label)
            predicted_labels = viterbi_decode(sequence,
                                              label_to_words_to_counts,
                                              label_denoms,
                                              label_unigrams,
                                              label_bigrams)
            output_tuples = list(zip(sequence, predicted_labels))
            output_json = json.dumps(output_tuples)
            results.write(output_json + '\n')
    results.close()


def viterbi_decode(sequence, label_to_words_to_counts, label_denoms,
        label_unigrams, label_bigrams):
    # viterbi data structure
    viterbi = defaultdict(dict)
    uni_denom = sum(label_unigrams.values()) - label_unigrams[(cfg.START,)]

    for i in range(len(sequence)):
        # loop through all labels
        for label in label_to_words_to_counts:
            log_max_transition = float('-inf')
            back_pointer = None

            # if it's the first
            if i == 0:
                back_pointer = cfg.START
                log_max_transition = get_log_prob((cfg.START, label),
                                                  label_unigrams,
                                                  label_bigrams,
                                                  uni_denom)
            else:
                for prev_label in viterbi[i - 1]:
                    log_transition = get_log_prob((prev_label, label),
                                                  label_unigrams,
                                                  label_bigrams,
                                                  uni_denom)
                    log_transition += viterbi[i - 1][prev_label][0]

                    if log_transition >= log_max_transition:
                        log_max_transition = log_transition
                        back_pointer = prev_label

            # stop probability calculation
            if i == len(sequence) - 1:
                log_stop_prob = get_log_prob((label, cfg.STOP),
                                             label_unigrams,
                                             label_bigrams,
                                             uni_denom)
            else:
                # if we're not at the end, we don't want to include stop
                log_stop_prob = 0

            # getting the emission probability
            if sequence[i] not in label_to_words_to_counts[label]:
                # it's an UNK
                emission_numer = label_to_words_to_counts[label][cfg.UNK]
            else:
                emission_numer = label_to_words_to_counts[label][sequence[i]]
            emission_prob = emission_numer / label_denoms[label]
            if emission_prob == 0:
                log_emission = float('-inf')
            else:
                log_emission = math.log(emission_prob, 2)
            viterbi[i][label] = (log_stop_prob + log_emission + log_max_transition, back_pointer)

    labels = []
    max_end_label = None
    max_end_prob = float('-inf')
    for label in viterbi[len(sequence) - 1]:
        if viterbi[len(sequence) - 1][label][0] >= max_end_prob:
            max_end_prob = viterbi[len(sequence) - 1][label][0]
            max_end_label = label

    labels.append(max_end_label)
    for i in reversed(range(1, len(sequence))):
        labels.insert(0, viterbi[i][labels[0]][1])

    return labels


def get_log_prob(bigram, label_unigrams, label_bigrams, uni_denom):
    '''Get the log probability of a label transition smoothed with linear
    interpolation.
    '''
    # unigram part
    uni_numer = label_unigrams[(bigram[1],)]
    unigram_part = cfg.LAMBDA_1 * uni_numer / uni_denom

    # bigram part
    bi_numer = label_bigrams[bigram]
    bi_denom = label_unigrams[(bigram[0],)]
    bigram_part = cfg.LAMBDA_2 * bi_numer / bi_denom

    prob = bigram_part + unigram_part
    log_prob = math.log(prob, 2)

    return log_prob


def train_hmm(filepath):
    print('training bigram HMM on', filepath, '. . .')
    print('UNK_ADD_K is', cfg.UNK_ADD_K)
    labels_to_words_to_counts = defaultdict(Counter)
    label_denoms = {}
    label_unigrams = Counter()
    label_bigrams = Counter()
    #word_counts = Counter()

    with open(filepath) as f:
        for line in f:
            tuples = json.loads(line)
            labels = [cfg.START]
            for word, label in tuples:
                labels_to_words_to_counts[label][word] += 1
                labels.append(label)
                #word_counts[word] += 1
            labels.append(cfg.STOP)
            # add label counts
            add_n_gram_counts(1, label_unigrams, labels)
            add_n_gram_counts(2, label_bigrams, labels)

    # old UNKing code
    #for label, unigrams in labels_to_words_to_counts.items():
    #    # the set of all unigrams that have a count less than UNK_THRESHOLD
    #    unks = set()
    #    num_unks = 0
    #    for unigram, count in unigrams.items():
    #        if word_counts[unigram] < cfg.UNK_THRESHOLD:
    #            unks.add(unigram)
    #            num_unks += count

    #    for word in unks:
    #        del unigrams[word]

    #    unigrams[cfg.UNK] = num_unks

    # sanity checking that every label has some UNK probability
    for label, unigrams in labels_to_words_to_counts.items():
        unigrams[cfg.UNK] += cfg.UNK_ADD_K

    # pre-computing label denoms
    for label in labels_to_words_to_counts:
        label_denoms[label] = sum(labels_to_words_to_counts[label].values())

    return labels_to_words_to_counts, label_denoms, label_unigrams, label_bigrams


def add_n_gram_counts(n, n_grams, tokens):
    '''Adds the n-grams to the specified Counter from the specified tokens.'''
    for i in range(len(tokens) - (n - 1)):
        n_grams[tuple(tokens[i:i+n])] += 1

    return n_grams


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', help='Output predictions on test file.',
            action='store_true')
    parser.add_argument('--tune', help='Grid search over hyper-parameters.',
            action='store_true')
    args = parser.parse_args()
    main(args)
