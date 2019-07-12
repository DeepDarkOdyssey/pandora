import numpy as np
import time
import os
import requests
from typing import Iterable, TextIO, List, Tuple, Optional, Callable, NoReturn
from random import randint, choice, shuffle, seed
from .utils import jaccard_similarity, generate_hash_fn, reproducible_randoms
from .min_hash import MinHash, build_signature_matrix
from .lsh import LSH


def get_brown_corpus(
    url: str = "http://www.sls.hawaii.edu/bley-vroman/brown_nolines.txt",
    save_dir: str = "tmp",
) -> TextIO:
    if not os.path.exists(save_dir):
        os.mkdir("tmp")
    if os.path.exists(os.path.join(save_dir, "brown_corpus.txt")):
        print("Brown corpus already exists!")
    else:
        print("Downloading brown corpus...")
        r = requests.get("http://www.sls.hawaii.edu/bley-vroman/brown_nolines.txt")
        with open(os.path.join(save_dir, "brown_corpus.txt"), "w") as f:
            f.write(r.text)
        print(f"Brown corpus has been saved to {save_dir}")
    brown = open(os.path.join(save_dir, "brown_corpus.txt"))
    return brown


def brown_generate(text_line_reader: Iterable):
    paragraph = []
    for line in text_line_reader:
        if line.strip() == "":
            sample = "".join(paragraph)
            paragraph = []
            if not sample.startswith("#") and len(sample.split()) > 10:
                yield sample.strip()
        paragraph.append(line)


def build_test_set(
    data_gen: Iterable[str], max_samples: int, random_seed=0
) -> List[Tuple[str, int]]:

    seed(random_seed)
    samples = []
    class_id = 0
    for data in data_gen:
        for _ in range(randint(1, max_samples)):
            tokens = data.split()
            effective_length = len([len(token) > 0 for token in tokens])
            num_replacement = effective_length // 10
            for __ in range(num_replacement):
                id2replace = choice(range(len(tokens)))
                cloned_token = choice(tokens)
                tokens[id2replace] = cloned_token
            samples.append((" ".join(tokens), class_id))
        class_id += 1

    shuffle(samples)
    return samples


def calculate_metrics(
    true_labels: List[int], pred_labels: List[int]
) -> Tuple[float, float]:

    true_positives, true_negatives, false_positives, false_negatives = 0, 0, 0, 0
    for true_label, pred_label in zip(true_labels, pred_labels):
        if true_label == pred_label:
            if true_label == 1:
                true_positives += 1
            else:
                true_negatives += 1
        else:
            if pred_label == 1:
                false_positives += 1
            else:
                false_negatives += 1
    assert true_positives + true_negatives + false_positives + false_negatives == len(
        true_labels
    )
    false_positive_rate = false_positives / (false_positives + true_negatives)
    false_negative_rate = false_negatives / (false_negatives + true_positives)
    # return true_positives, true_negatives, false_positives, false_negatives
    return false_positive_rate, false_negative_rate


def minhash_test_one(
    tokenizer: Optional[Callable[[str], List[str]]] = None,
    sig_matrix_path: str = "./tmp/signature_matrix.npy",
    threshold: float = 0.6,
    ngrams: int = 1,
    permutation_way: str = "perm",
    signature_len: int = 200,
    max_samples: int = 10,
    random_seed: int = 0,
) -> NoReturn:
    print("building test set...")
    tick = time.time()
    data_gen = brown_generate(get_brown_corpus())
    samples = build_test_set(data_gen, max_samples, random_seed)
    tock = time.time()
    print(f"Time cost by generating samples: {tock-tick:.2f}s")
    print(f"Total samples: {len(samples)}")

    if not tokenizer:
        tokenizer = lambda string: string.strip().replace("\n", " ").split()

    if os.path.exists(sig_matrix_path):
        signature_matrix = np.load(sig_matrix_path, allow_pickle=True)
    else:
        print("Signature matrix not exists!")
        signature_matrix = build_signature_matrix(
            samples=samples,
            tokenizer=tokenizer,
            output_path=sig_matrix_path,
            ngrams=ngrams,
            permutation_way=permutation_way,
            signature_len=signature_len,
            random_seed=random_seed,
        )
    print("Signature matrix has been loaded")

    seed(os.urandom(10))
    chosen_sample = choice(samples)
    chosen_min_hash = MinHash(
        tokens=tokenizer(chosen_sample[0]),
        ngrams=ngrams,
        permutation_way=permutation_way,
        random_seed=random_seed,
        signature_len=signature_len,
    )

    true_labels = [1 if chosen_sample[1] == sample[1] else 0 for sample in samples]
    pred_labels = [0] * len(samples)
    assert len(true_labels) == len(pred_labels)

    print("Finding near duplicates...")
    duplicates = chosen_min_hash.find_near_duplicates(signature_matrix, threshold)
    print(f"Duplicates found :\n{duplicates}")
    for sample_id, _ in duplicates:
        pred_labels[sample_id] = 1
    fpr, fnr = calculate_metrics(true_labels, pred_labels)
    print(f"False postive rate: {fpr}, false negative rate: {fnr}")


def minhash_test_all(
    tokenizer: Optional[Callable[[str], List[str]]] = None,
    sig_matrix_path: str = "./tmp/signature_matrix.npy",
    threshold: float = 0.6,
    ngrams: int = 1,
    permutation_way: str = "perm",
    signature_len: int = 200,
    max_samples: int = 10,
    random_seed: int = 0,
) -> NoReturn:
    print("building test set...")
    tick = time.time()
    data_gen = brown_generate(get_brown_corpus())
    samples = build_test_set(data_gen, max_samples, random_seed)
    tock = time.time()
    print(f"Time cost by generating samples: {tock-tick:.2f}s")
    print(f"Total samples: {len(samples)}")

    if not tokenizer:
        tokenizer = lambda string: string.strip().replace("\n", " ").split()

    if os.path.exists(sig_matrix_path):
        signature_matrix = np.load(sig_matrix_path, allow_pickle=True)
    else:
        print("Signature matrix not exists!")
        signature_matrix = build_signature_matrix(
            samples=samples,
            tokenizer=tokenizer,
            output_path=sig_matrix_path,
            ngrams=ngrams,
            permutation_way=permutation_way,
            signature_len=signature_len,
            random_seed=random_seed,
        )
    print("Signature matrix has been loaded")

    print("Finding duplicates for each sample...")
    avg_fpr, avg_fnr = 0, 0
    for sample_id, sample in enumerate(samples):
        true_labels = [1 if sample[1] == sample_[1] else 0 for sample_ in samples]
        pred_labels = [0] * len(samples)
        assert len(true_labels) == len(pred_labels)

        min_hash = MinHash(
            tokens=tokenizer(sample[0]),
            ngrams=ngrams,
            permutation_way=permutation_way,
            random_seed=random_seed,
            signature_len=signature_len,
        )

        if sample_id % 100 == 0:
            checkpoint_time = time.time()
            print(
                f"\r{sample_id}/{len(samples)}----"
                f"total time: {checkpoint_time - tick:.2f}s----"
                f"avg time: {(checkpoint_time - tick) / (sample_id + 1):.4f}s",
                end="",
            )

        duplicates = min_hash.find_near_duplicates(signature_matrix, 0.6)
        for i, _ in duplicates:
            pred_labels[i] = 1
        fpr, fnr = calculate_metrics(true_labels, pred_labels)
        avg_fpr += fpr
        avg_fnr += fnr

    tock = time.time()
    print()
    print(f"Finished! Time cost: {tock - tick:.2f}s")
    print(f"Average FPR: {avg_fpr/len(samples)}, average FNR: {avg_fnr / len(sample)}")


def lsh_test_one(
    tokenizer: Optional[Callable[[str], List[str]]] = None,
    sig_matrix_path: str = "./tmp/signature_matrix.npy",
    threshold: int = 1,
    ngrams: int = 1,
    permutation_way: str = "perm",
    signature_len: int = 200,
    max_samples: int = 10,
    random_seed: int = 0,
    num_bands: Optional[int] = 20,
    num_cols: Optional[int] = None,
    num_buckets: int = 20
):
    print("building test set...")
    tick = time.time()
    data_gen = brown_generate(get_brown_corpus())
    samples = build_test_set(data_gen, max_samples, random_seed)
    tock = time.time()
    print(f"Time cost by generating samples: {tock-tick:.2f}s")
    print(f"Total samples: {len(samples)}")

    if not tokenizer:
        tokenizer = lambda string: string.strip().replace("\n", " ").split()
    if os.path.exists(sig_matrix_path):
        signature_matrix = np.load(sig_matrix_path, allow_pickle=True)
    else:
        print("Signature matrix not exists!")
        signature_matrix = build_signature_matrix(
            samples=samples,
            tokenizer=tokenizer,
            output_path=sig_matrix_path,
            ngrams=ngrams,
            permutation_way=permutation_way,
            signature_len=signature_len,
            random_seed=random_seed,
        )
    print("Signature matrix has been loaded")

    seed(os.urandom(10))
    chosen_sample = choice(samples)
    chosen_min_hash = MinHash(
        tokens=tokenizer(chosen_sample[0]),
        ngrams=ngrams,
        permutation_way=permutation_way,
        random_seed=random_seed,
        signature_len=signature_len,
    )

    lsh = LSH(signature_matrix, num_bands, num_cols, num_buckets, seed)
    print(lsh.get_candidates(chosen_min_hash.signature, threshold))


if __name__ == "__main__":
    lsh_test_one()
