from FastAutoAugment.nlp_augmentations import *

import pickle
import wandb

def clean_augmented_data(augmented_data):
    return_val = []
    for data_point in augmented_data:
        if len(data_point.strip()) == 0:
            continue
        return_val.append(data_point)
    return return_val

def create_augmented_dataset(X, y, f, p, n):
    new_x = []
    new_y = []
    for x_elem, y_elem in zip(X, y):
        augmented_data = f(x_elem, p, n)
        augmented_data = clean_augmented_data(augmented_data)
        new_x += augmented_data
        new_y += [y_elem] * len(augmented_data)
    assert len(new_x) == len(new_y)
    return new_x, new_y
            

if __name__ == "__main__":
    full_val_data = pickle.load(open('data/yahoo_answers_v1/yahoo_answers_full_val.pkl', 'rb'))
    X = full_val_data['X'][:3]
    y = full_val_data['y'][:3]
    X_aug, y_aug = create_augmented_dataset(X, y, random_insertion_transform, 0.3, 5)
    print(X_aug)
    print(y_aug)
