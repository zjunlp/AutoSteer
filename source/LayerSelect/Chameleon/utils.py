import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class SafetyAwareness:
    def __init__(self, pos_npy_path, neg_npy_path):
        """
        Initialize: Load two .npy files, each with shape [N, D]
        """
        self.pos_list = np.load(pos_npy_path, mmap_mode='r')
        self.neg_list = np.load(neg_npy_path, mmap_mode='r')

        assert self.pos_list.shape == self.neg_list.shape, "Positive and Negative embedding sizes must match."

        self.random_selected_pos_list = None
        self.random_selected_neg_list = None
        self.selected = False

    def random_select(self, sample_num):
        """
        Randomly select sample_num indices (synchronously for pos and neg)
        """
        total = sample_num
        assert total <= len(self.pos_list), "Sample number exceeds available data."

        random_indices = np.random.choice(len(self.pos_list), size=total, replace=False)

        self.random_selected_pos_list = self.pos_list[random_indices]
        self.random_selected_neg_list = self.neg_list[random_indices]

        self.selected = True

    def calculate_safetyAwareness(self):
        """
        Compute the safety awareness score (average cosine similarity)
        """
        if not self.selected:
            raise RuntimeError("You must run random_select() before calling calculate_safetyAwareness()")

        # Get CAAs
        self.CAA_array = self.random_selected_pos_list - self.random_selected_neg_list

        # Compute cosine similarity matrix
        self.cosine_sim_matrix = cosine_similarity(self.CAA_array)

        # Average cosine similarity
        num_vectors = self.CAA_array.shape[0]
        upper_tri_indices = np.triu_indices(num_vectors, k=1)
        self.avg_cosine_similarity = np.mean(self.cosine_sim_matrix[upper_tri_indices])

        return self.avg_cosine_similarity