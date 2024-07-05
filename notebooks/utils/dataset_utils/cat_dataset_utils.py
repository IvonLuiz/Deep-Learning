import os
import pickle
import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import tensorflow as tf

class CIFAR10Dataset:
    def __init__(self, cifar10_path, classification_type='binary', num_train_exem=3000, num_test_exem=600):
        """
        Initialize the CIFAR10CatClassifier.

        Parameters:
        cifar10_path (str): Path to the CIFAR-10 dataset.
        classification_type (str): Type of classification ('binary' or 'multiclass').
        num_train_exem (int): Number of training examples to use.
        num_test_exem (int): Number of test examples to use.
        """
        self.cifar10_path = cifar10_path
        self.classification_type = classification_type
        self.num_train_exem = num_train_exem
        self.num_test_exem = num_test_exem
        self.load_data()
        self.process_data()
        
    def unpickle(self, file):
        with open(file, 'rb') as fo:
            data_dict = pickle.load(fo, encoding='bytes')
        return data_dict
    
    def load_data(self):
        # Load all training batches
        train_data = []
        train_labels = []
        for i in range(1, 6):
            batch = self.unpickle(os.path.join(self.cifar10_path, f'data_batch_{i}'))
            train_data.append(batch[b'data'])
            train_labels.append(batch[b'labels'])

        self.train_data = np.concatenate(train_data)
        self.train_labels = np.concatenate(train_labels)

        # Load test batch
        test_batch = self.unpickle(os.path.join(self.cifar10_path, 'test_batch'))
        self.test_data = test_batch[b'data']
        self.test_labels = np.array(test_batch[b'labels'])

        # Load label names
        meta = self.unpickle(os.path.join(self.cifar10_path, 'batches.meta'))
        self.label_names = [label.decode('utf-8') for label in meta[b'label_names']]
    
    def process_data(self):
        if self.classification_type == 'binary':
            self.process_binary_classification()
        elif self.classification_type == 'multiclass':
            self.process_multiclass_classification()
        else:
            raise ValueError("Invalid classification type. Choose 'binary' or 'multiclass'.")
    
    def process_binary_classification(self):
        # Define the label for the "cat" class
        cat_label = self.label_names.index('cat')

        # Separate cat images from other images in the training set
        train_cat_indices = np.where(self.train_labels == cat_label)[0]
        train_other_indices = np.where(self.train_labels != cat_label)[0]

        self.train_cat_data = self.train_data[train_cat_indices]
        self.train_other_data = self.train_data[train_other_indices]

        self.train_cat_labels = np.ones(train_cat_indices.shape[0])  # Label 1 for cats
        self.train_other_labels = np.zeros(train_other_indices.shape[0])  # Label 0 for other classes

        # Repeat for the test set
        test_cat_indices = np.where(self.test_labels == cat_label)[0]
        test_other_indices = np.where(self.test_labels != cat_label)[0]

        self.test_cat_data = self.test_data[test_cat_indices]
        self.test_other_data = self.test_data[test_other_indices]

        self.test_cat_labels = np.ones(test_cat_indices.shape[0])  # Label 1 for cats
        self.test_other_labels = np.zeros(test_other_indices.shape[0])  # Label 0 for other classes

        train_class_examples = int(self.num_train_exem / 2)
        test_class_examples = int(self.num_test_exem / 2)

        self.train_cat_data, self.train_cat_labels = shuffle(self.train_cat_data, self.train_cat_labels, random_state=0)
        self.train_other_data, self.train_other_labels = shuffle(self.train_other_data, self.train_other_labels, random_state=0)
        self.test_cat_data, self.test_cat_labels = shuffle(self.test_cat_data, self.test_cat_labels, random_state=0)
        self.test_other_data, self.test_other_labels = shuffle(self.test_other_data, self.test_other_labels, random_state=0)

        self.train_data = np.vstack((self.train_cat_data[:train_class_examples], self.train_other_data[:train_class_examples]))
        self.train_labels = np.concatenate((self.train_cat_labels[:train_class_examples], self.train_other_labels[:train_class_examples]))

        self.test_data = np.vstack((self.test_cat_data[:test_class_examples], self.test_other_data[:test_class_examples]))
        self.test_labels = np.concatenate((self.test_cat_labels[:test_class_examples], self.test_other_labels[:test_class_examples]))

        self.train_data, self.train_labels = shuffle(self.train_data, self.train_labels, random_state=0)
        self.test_data, self.test_labels = shuffle(self.test_data, self.test_labels, random_state=0)
    
    def process_multiclass_classification(self):
        train_class_examples = self.num_train_exem // len(self.label_names)
        test_class_examples = self.num_test_exem // len(self.label_names)

        train_data_split = []
        train_labels_split = []
        test_data_split = []
        test_labels_split = []

        for i, label_name in enumerate(self.label_names):
            label_indices = np.where(self.train_labels == i)[0]
            train_data_split.append(self.train_data[label_indices][:train_class_examples])
            train_labels_split.append(np.full(train_class_examples, i))
            
            label_indices = np.where(self.test_labels == i)[0]
            test_data_split.append(self.test_data[label_indices][:test_class_examples])
            test_labels_split.append(np.full(test_class_examples, i))

        self.train_data = np.vstack(train_data_split)
        self.train_labels = np.concatenate(train_labels_split)
        self.test_data = np.vstack(test_data_split)
        self.test_labels = np.concatenate(test_labels_split)

        self.train_data, self.train_labels = shuffle(self.train_data, self.train_labels, random_state=0)
        self.test_data, self.test_labels = shuffle(self.test_data, self.test_labels, random_state=0)
    
    def plot_image(self, data, indices):
        num_images = len(indices)
        fig, axes = plt.subplots(1, num_images, figsize=(15, 3))

        for i, idx in enumerate(indices):
            img = data[idx].reshape(3, 32, 32).transpose(1, 2, 0)
            axes[i].imshow(img)
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()

    def plot_cat_and_other_images(self, num_images):
        if self.classification_type != 'binary':
            raise ValueError("plot_cat_and_other_images is only available for binary classification.")
        
        # Plot cat images
        print("Cat Images:")
        self.plot_image(self.train_cat_data, range(num_images))

        # Plot other images
        print("Other Images:")
        self.plot_image(self.train_other_data, range(num_images))

    def plot_random_images(self, num_images, data_type='train', label='all'):
        if data_type == 'train':
            data = self.train_data
            labels = self.train_labels
        elif data_type == 'test':
            data = self.test_data
            labels = self.test_labels
        else:
            raise ValueError("Invalid data type. Choose 'train' or 'test'.")

        if label == 'cat':
            indices = np.where(labels == 1)[0] if self.classification_type == 'binary' else np.where(labels == self.label_names.index('cat'))[0]
        elif label == 'other':
            indices = np.where(labels == 0)[0] if self.classification_type == 'binary' else np.where(labels != self.label_names.index('cat'))[0]
        elif label == 'all':
            indices = np.arange(len(labels))
        else:
            raise ValueError("Invalid label. Choose 'cat', 'other', or 'all'.")

        random_indices = np.random.choice(indices, num_images, replace=False)
        
        fig, axes = plt.subplots(1, num_images, figsize=(15, 3))

        for i, idx in enumerate(random_indices):
            img = data[idx].reshape(3, 32, 32).transpose(1, 2, 0)
            axes[i].imshow(img)
            axes[i].axis('off')
            if self.classification_type == 'binary':
                axes[i].set_title('Cat' if labels[idx] == 1 else 'Other')
            else:
                axes[i].set_title(self.label_names[labels[idx]])

        plt.tight_layout()
        plt.show()

    
    def get_train_data(self, as_dataset=False):
        if as_dataset:
            train_data_ds = tf.data.Dataset.from_tensor_slices(self.train_data)
            train_labels_ds = tf.data.Dataset.from_tensor_slices(self.train_labels)
            return train_data_ds, train_labels_ds
        return self.train_data, self.train_labels


    def get_test_data(self, as_dataset=False):
        if as_dataset:
            test_data_ds = tf.data.Dataset.from_tensor_slices(self.test_data)
            test_labels_ds = tf.data.Dataset.from_tensor_slices(self.test_labels)
            return test_data_ds, test_labels_ds
        return self.test_data, self.test_labels
