#!/usr/bin/env python3

import argparse
import json
from collections import Counter
from collections import defaultdict

import config as cfg


def main(args):
    if args.tri:
        evaluate(cfg.tri_train_out, cfg.small_train_path)
        evaluate(cfg.tri_dev_out, cfg.dev_path)
        if args.test:
            evaluate(cfg.tri_test_out, cfg.test_path)
    else:
        evaluate(cfg.train_out, cfg.train_path)
        evaluate(cfg.dev_out, cfg.dev_path)
        if args.test:
            evaluate(cfg.test_out, cfg.test_path)


def evaluate(predictions, gold):
    print()
    print('evaluating predictions at', predictions, 'against gold at', gold)
    predict_file = open(predictions)
    correct_file = open(gold)

    num_correct = 0
    total = 0
    label_counts = Counter()
    label_correct = Counter()
    label_percents = {}
    pred_labels = set()

    per_label_predictions = defaultdict(Counter)

    for predictions, correct in zip(predict_file, correct_file):
        pred_tuples = json.loads(predictions)
        corr_tuples = json.loads(correct)
        for i in range(len(pred_tuples)):
            total += 1
            label_counts[corr_tuples[i][1]] += 1
            if pred_tuples[i] == corr_tuples[i]:
                num_correct += 1
                label_correct[corr_tuples[i][1]] += 1

            y = corr_tuples[i][1]
            y_hat = pred_tuples[i][1]
            pred_labels.add(y_hat)
            per_label_predictions[y][y_hat] += 1

    for label in label_counts:
        label_percents[label] = label_correct[label] / float(label_counts[label])

    accuracy = num_correct / total

    print('accuracy:', accuracy)
    print(len(label_percents), 'total labels')
    print('percentage correct per label:', label_percents)

    print_confusion_matrix(per_label_predictions, sorted(pred_labels.union(label_correct.keys())))

    return accuracy


def print_confusion_matrix(per_label_predictions, labels, f=None):
    print('generating confusion matrix')
    for label in labels:
        print(label, end=' ', file=f)
    print('', file=f)
    for label in labels:
        predictions = per_label_predictions[label]
        print(label, end=' & ', file=f)
        for label in labels:
            count = predictions[label]
            print(count, end=' & ', file=f)
        print('', file=f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tri', help='Output predictions on test file.',
            action='store_true')
    parser.add_argument('--test', help='Output predictions on test file.',
            action='store_true')
    args = parser.parse_args()
    main(args)
