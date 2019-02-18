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
        csv_lambda_3s = []
        csv_train_accs = []
        csv_dev_accs = []

        K_vals = [10**exp for exp in range(-5, 3)]
        for K in K_vals:
            print('tuning trigram HMM . . .')
            print()
            cfg.T_UNK_ADD_K = K
            labels_to_words_to_counts, label_denoms, label_unigrams, label_bigrams, \
                    label_trigrams = train_hmm(cfg.train_path)
            print()
            for lambda1 in cfg.T_LAMBDA_1s:
                for lambda2 in cfg.T_LAMBDA_2s:
                    if lambda1 + lambda2 > 0.9:
                        continue
                    lambda3 = 1 - lambda1 - lambda2
                    csv_K_vals.append(cfg.T_UNK_ADD_K)
                    cfg.T_LAMBDA_1 = lambda1
                    csv_lambda_1s.append(cfg.T_LAMBDA_1)
                    cfg.T_LAMBDA_2 = lambda2
                    csv_lambda_2s.append(cfg.T_LAMBDA_2)
                    cfg.T_LAMBDA_3 = lambda3
                    csv_lambda_3s.append(cfg.T_LAMBDA_3)
                    predict(cfg.train_path, cfg.tri_train_out, labels_to_words_to_counts,
                            label_denoms, label_unigrams, label_bigrams, label_trigrams)
                    train_acc = calculate_accuracies.evaluate(cfg.tri_train_out, cfg.train_path)
                    csv_train_accs.append(train_acc)
                    predict(cfg.dev_path, cfg.tri_dev_out, labels_to_words_to_counts,
                            label_denoms, label_unigrams, label_bigrams, label_trigrams)
                    dev_acc = calculate_accuracies.evaluate(cfg.tri_dev_out, cfg.dev_path)
                    csv_dev_accs.append(dev_acc)
                    print()
                    if args.test:
                        predict(cfg.test_path, cfg.tri_test_out, labels_to_words_to_counts,
                                label_denoms, label_unigrams, label_bigrams, label_trigrams)
                        calculate_accuracies.evaluate(cfg.tri_test_out, cfg.test_path)

        pd.DataFrame({'unk add-K': csv_K_vals,
                      'lambda 1': csv_lambda_1s,
                      'lambda 2': csv_lambda_2s,
                      'lambda 3': csv_lambda_3s,
                      'train accuracy': csv_train_accs,
                      'dev accuracy': csv_dev_accs}).to_csv('tuning-tri-hmm.csv',
                                                            index=False)
    else:
        labels_to_words_to_counts, label_denoms, label_unigrams, label_bigrams, \
                label_trigrams = train_hmm(cfg.train_path)
        predict(cfg.train_path, cfg.tri_train_out, labels_to_words_to_counts,
                label_denoms, label_unigrams, label_bigrams, label_trigrams)
        calculate_accuracies.evaluate(cfg.tri_train_out, cfg.train_path)
        predict(cfg.dev_path, cfg.tri_dev_out, labels_to_words_to_counts, \
                label_denoms, label_unigrams, label_bigrams, label_trigrams)
        calculate_accuracies.evaluate(cfg.tri_dev_out, cfg.dev_path)
        if args.test:
            predict(cfg.test_path, cfg.tri_test_out, labels_to_words_to_counts, \
                    label_denoms, label_unigrams, label_bigrams, label_trigrams)
            calculate_accuracies.evaluate(cfg.tri_test_out, cfg.test_path)


def predict(input_path, output_path, labels_to_words_to_counts, label_denoms,
        label_unigrams, label_bigrams, label_trigrams):
    print('tagging', input_path, 'and saving to', output_path, '. . .')
    print('LAMBDA_1:', cfg.T_LAMBDA_1, 'LAMBDA_2:', cfg.T_LAMBDA_2,
            'LAMBDA_3', cfg.T_LAMBDA_3)
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
                                              labels_to_words_to_counts,
                                              label_denoms,
                                              label_unigrams,
                                              label_bigrams,
                                              label_trigrams)
            output_tuples = list(zip(sequence, predicted_labels))
            output_json = json.dumps(output_tuples)
            results.write(output_json + '\n')
    results.close()


def viterbi_decode(sequence, labels_to_words_to_counts, label_denoms,
        label_unigrams, label_bigrams, label_trigrams):
    # viterbi data structure
    viterbi = defaultdict(lambda: defaultdict(dict))
    uni_denom = sum(label_unigrams.values()) - label_unigrams[(cfg.START,)]

    for i in range(len(sequence)):
        # label2 represents the most recent label
        for label2 in labels_to_words_to_counts:
            log_max_transition = float('-inf')
            back_pointer = None
            if i == 0:
                # if we're on the first token
                back_pointer = (cfg.START, cfg.START)
                log_max_transition = get_log_prob((cfg.START, cfg.START, label2),
                                                  label_unigrams,
                                                  label_bigrams,
                                                  label_trigrams,
                                                  uni_denom)
                log_emission = get_emission_log_prob(label2, sequence[i],
                        labels_to_words_to_counts, label_denoms)
                viterbi[i][cfg.START][label2] = \
                        (log_emission + log_max_transition, back_pointer)
            elif i == 1:
                # if we're on the second token
                for prev_label in labels_to_words_to_counts:
                    back_pointer = (cfg.START, prev_label)
                    log_transition = get_log_prob((cfg.START, prev_label, label2),
                                                  label_unigrams,
                                                  label_bigrams,
                                                  label_trigrams,
                                                  uni_denom)
                    log_transition += viterbi[i - 1][cfg.START][prev_label][0]
                    log_max_transition = log_transition
                    log_emission = get_emission_log_prob(label2, sequence[i],
                            labels_to_words_to_counts, label_denoms)
                    viterbi[i][prev_label][label2] = \
                            (log_emission + log_max_transition, back_pointer)
            else:
                for label1 in labels_to_words_to_counts:
                    for prev_label in viterbi[i - 1]:
                        log_transition = get_log_prob((prev_label, label1, label2),
                                                      label_unigrams,
                                                      label_bigrams,
                                                      label_trigrams,
                                                      uni_denom)
                        log_transition += viterbi[i - 1][prev_label][label1][0]
                        if log_transition >= log_max_transition:
                            log_max_transition = log_transition
                            back_pointer = (prev_label, label1)
                    # stop probability calculation
                    if i == len(sequence) - 1:
                        log_stop_prob = get_log_prob((label1, label2, cfg.STOP),
                                                     label_unigrams,
                                                     label_bigrams,
                                                     label_trigrams,
                                                     uni_denom)
                    else:
                        log_stop_prob = 0
                    # getting the emission probability
                    log_emission = get_emission_log_prob(label2, sequence[i],
                            labels_to_words_to_counts, label_denoms)
                    viterbi[i][label1][label2] = (log_stop_prob +
                            log_emission + log_max_transition, back_pointer)

    labels = []
    max_end_label = None
    max_end_prob = float('-inf')
    for label1 in viterbi[len(sequence) - 1]:
        for label2 in viterbi[len(sequence) - 1][label1]:
            if viterbi[len(sequence) - 1][label1][label2][0] >= max_end_prob:
                max_end_prob = viterbi[len(sequence) - 1][label1][label2][0]
                max_end_label = (label1, label2)

    labels.append(max_end_label)
    for i in reversed(range(1, len(sequence))):
        labels.insert(0, viterbi[i][labels[0][0]][labels[0][1]][1])

    ret = []
    for label in labels:
        ret.append(label[1])

    return ret


def get_emission_log_prob(label, word, labels_to_words_to_counts, label_denoms):
    # getting the emission probability
    if word not in labels_to_words_to_counts[label]:
        # it's an UNK
        emission_numer = labels_to_words_to_counts[label][cfg.UNK]
    else:
        emission_numer = labels_to_words_to_counts[label][word]
    emission_prob = emission_numer / label_denoms[label]
    if emission_prob == 0:
        log_emission = float('-inf')
    else:
        log_emission = math.log(emission_prob, 2)

    return log_emission


def get_log_prob(trigram, label_unigrams, label_bigrams, label_trigrams, uni_denom):
    '''Get the log probability of a label transition smoothed with linear
    interpolation.
    '''
    vocab_size = len(label_unigrams) - 1
    # unigram part
    uni_numer = label_unigrams[trigram[2:]]
    unigram_part = cfg.T_LAMBDA_1 * uni_numer / uni_denom

    # bigram part
    bi_numer = label_bigrams[trigram[1:]]
    bi_denom = label_unigrams[trigram[1:2]]
    bigram_part = cfg.T_LAMBDA_2 * bi_numer / bi_denom

    # trigram part
    tri_numer = label_trigrams[trigram]
    tri_denom = label_bigrams[trigram[:2]]
    if tri_denom == 0:
        vocab_size = len(label_unigrams) - 1
        trigram_part = cfg.T_LAMBDA_3 * 1 / vocab_size
    else:
        trigram_part = cfg.T_LAMBDA_3 * tri_numer / tri_denom

    prob = unigram_part + bigram_part + trigram_part
    log_prob = math.log(prob, 2)

    return log_prob


def train_hmm(filepath):
    print('training trigram HMM on', filepath, '. . .')
    print('UNK_ADD_K is', cfg.T_UNK_ADD_K)
    labels_to_words_to_counts = defaultdict(Counter)
    label_denoms = {}
    label_unigrams = Counter()
    label_bigrams = Counter()
    label_trigrams = Counter()
    #word_counts = Counter()

    with open(filepath) as f:
        for line in f:
            tuples = json.loads(line)
            labels = [cfg.START, cfg.START]
            for word, label in tuples:
                labels_to_words_to_counts[label][word] += 1
                labels.append(label)
                #word_counts[word] += 1
            labels.append(cfg.STOP)
            # add label counts
            add_n_gram_counts(1, label_unigrams, labels)
            add_n_gram_counts(2, label_bigrams, labels)
            add_n_gram_counts(3, label_trigrams, labels)

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
        unigrams[cfg.UNK] += cfg.T_UNK_ADD_K

    # pre-computing label denoms
    for label in labels_to_words_to_counts:
        label_denoms[label] = sum(labels_to_words_to_counts[label].values())

    return labels_to_words_to_counts, label_denoms, label_unigrams, \
            label_bigrams, label_trigrams


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
