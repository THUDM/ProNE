# encoding=utf8
import numpy as np
import networkx as nx

import scipy.sparse
import scipy.sparse as sp
from scipy import linalg
from scipy.special import iv

from sklearn import preprocessing
from sklearn.utils.extmath import randomized_svd

import argparse
import time


class ProNE():
	def __init__(self, graph_file, emb_file1, emb_file2, dimension):
		self.graph = graph_file
		self.emb1 = emb_file1
		self.emb2 = emb_file2
		self.dimension = dimension

		self.G = nx.read_edgelist(self.graph, nodetype=int, create_using=nx.DiGraph())
		self.G = self.G.to_undirected()
		self.node_number = self.G.number_of_nodes()
		matrix0 = scipy.sparse.lil_matrix((self.node_number, self.node_number))

		for e in self.G.edges():
			if e[0] != e[1]:
				matrix0[e[0], e[1]] = 1
				matrix0[e[1], e[0]] = 1
		self.matrix0 = scipy.sparse.csr_matrix(matrix0)
		print(matrix0.shape)

	def get_embedding_rand(self, matrix):
		# Sparse randomized tSVD for fast embedding
		t1 = time.time()
		l = matrix.shape[0]
		smat = scipy.sparse.csc_matrix(matrix)  # convert to sparse CSC format
		print('svd sparse', smat.data.shape[0] * 1.0 / l ** 2)
		U, Sigma, VT = randomized_svd(smat, n_components=self.dimension, n_iter=5, random_state=None)
		U = U * np.sqrt(Sigma)
		U = preprocessing.normalize(U, "l2")
		print('sparsesvd time', time.time() - t1)
		return U

	def get_embedding_dense(self, matrix, dimension):
		# get dense embedding via SVD
		t1 = time.time()
		U, s, Vh = linalg.svd(matrix, full_matrices=False, check_finite=False, overwrite_a=True)
		U = np.array(U)
		U = U[:, :dimension]
		s = s[:dimension]
		s = np.sqrt(s)
		U = U * s
		U = preprocessing.normalize(U, "l2")
		print('densesvd time', time.time() - t1)
		return U

	def pre_factorization(self, tran, mask):
		# Network Embedding as Sparse Matrix Factorization
		t1 = time.time()
		l1 = 0.75
		C1 = preprocessing.normalize(tran, "l1")
		neg = np.array(C1.sum(axis=0))[0] ** l1

		neg = neg / neg.sum()

		neg = scipy.sparse.diags(neg, format="csr")
		neg = mask.dot(neg)
		print("neg", time.time() - t1)

		C1.data[C1.data <= 0] = 1
		neg.data[neg.data <= 0] = 1

		C1.data = np.log(C1.data)
		neg.data = np.log(neg.data)

		C1 -= neg
		F = C1
		features_matrix = self.get_embedding_rand(F)
		return features_matrix

	def chebyshev_gaussian(self, A, a, order=10, mu=0.5, s=0.5):
		# NE Enhancement via Spectral Propagation
		print('Chebyshev Series -----------------')
		t1 = time.time()

		if order == 1:
			return a

		A = sp.eye(self.node_number) + A
		DA = preprocessing.normalize(A, norm='l1')
		L = sp.eye(self.node_number) - DA

		M = L - mu * sp.eye(self.node_number)

		Lx0 = a
		Lx1 = M.dot(a)
		Lx1 = 0.5 * M.dot(Lx1) - a

		conv = iv(0, s) * Lx0
		conv -= 2 * iv(1, s) * Lx1
		for i in range(2, order):
			Lx2 = M.dot(Lx1)
			Lx2 = (M.dot(Lx2) - 2 * Lx1) - Lx0
			#         Lx2 = 2*L.dot(Lx1) - Lx0
			if i % 2 == 0:
				conv += 2 * iv(i, s) * Lx2
			else:
				conv -= 2 * iv(i, s) * Lx2
			Lx0 = Lx1
			Lx1 = Lx2
			del Lx2
			print('Bessell time', i, time.time() - t1)
		mm = A.dot(a - conv)
		emb = self.get_embedding_dense(mm, self.dimension)
		return emb



def save_embedding(emb_file, features):
	# save node embedding into emb_file with word2vec format
	f_emb = open(emb_file, 'w')
	f_emb.write(str(len(features)) + " " + str(features.shape[1]) + "\n")
	for i in range(len(features)):
		s = str(i) + " " + " ".join(str(f) for f in features[i].tolist())
		f_emb.write(s + "\n")
	f_emb.close()



def parse_args():
	parser = argparse.ArgumentParser(description="Run ProNE.")
	parser.add_argument('-graph', nargs='?', default='data/blogcatalog.ungraph',
						help='Graph path')
	parser.add_argument('-emb1', nargs='?', default='emb/blogcatalog.emb',
						help='Output path of sparse embeddings')
	parser.add_argument('-emb2', nargs='?', default='emb/blogcatalog_enhanced.emb',
						help='Output path of enhanced embeddings')
	parser.add_argument('-dimension', type=int, default=128,
						help='Number of dimensions. Default is 128.')
	parser.add_argument('-step', type=int, default=10,
						help='Step of recursion. Default is 10.')
	parser.add_argument('-theta', type=float, default=0.5,
						help='Parameter of ProNE. Default is 0.5.')
	parser.add_argument('-mu', type=float, default=0.2,
						help='Parameter of ProNE. Default is 0.2')
	return parser.parse_args()


def main():
	args = parse_args()

	t_0 = time.time()
	model = ProNE(args.graph, args.emb1, args.emb2, args.dimension)
	t_1 = time.time()

	features_matrix = model.pre_factorization(model.matrix0, model.matrix0)
	t_2 = time.time()

	embeddings_matrix = model.chebyshev_gaussian(model.matrix0, features_matrix, args.step, args.mu, args.theta)
	t_3 = time.time()


	print('---', model.node_number)
	print('total time', t_3 - t_0)
	print('sparse NE time', t_2 - t_1)
	print('spectral Pro time', t_3 - t_2)

	save_embedding(args.emb1, features_matrix)
	save_embedding(args.emb2, embeddings_matrix)
	print('save embedding done')


if __name__ == '__main__':
	main()