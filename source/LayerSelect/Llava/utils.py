import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class SafetyAwareness:
    def __init__(self, pos_npy_path, neg_npy_path):
        """
        初始化：加载两个 .npy 文件，每个文件形状为 [N, D]
        """
        self.pos_list = np.load(pos_npy_path, mmap_mode='r')
        self.neg_list = np.load(neg_npy_path, mmap_mode='r')

        assert self.pos_list.shape == self.neg_list.shape, "Positive and Negative embedding sizes must match."

        self.random_selected_pos_list = None
        self.random_selected_neg_list = None
        self.selected = False

    def random_select(self, sample_num):
        """
        随机选择 sample_num 个索引（对 pos 和 neg 同步采样）
        """
        total = sample_num
        assert total <= len(self.pos_list), "样本数量超过数据集长度"

        random_indices = np.random.choice(len(self.pos_list), size=total, replace=False)

        self.random_selected_pos_list = self.pos_list[random_indices]
        self.random_selected_neg_list = self.neg_list[random_indices]

        self.selected = True

    def calculate_safetyAwareness(self):
        """
        计算安全感知度（平均余弦相似度）
        """
        if not self.selected:
            raise RuntimeError("You must run random_select() before calling calculate_safetyAwareness()")

        # 差向量计算
        self.CAA_array = self.random_selected_pos_list - self.random_selected_neg_list

        # 计算余弦相似度矩阵
        self.cosine_sim_matrix = cosine_similarity(self.CAA_array)

        # 仅取非对角元素平均
        num_vectors = self.CAA_array.shape[0]
        upper_tri_indices = np.triu_indices(num_vectors, k=1)
        self.avg_cosine_similarity = np.mean(self.cosine_sim_matrix[upper_tri_indices])

        return self.avg_cosine_similarity