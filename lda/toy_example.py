import numpy as np
from scipy.stats import dirichlet
from sklearn.decomposition import LatentDirichletAllocation
# the symbol convention follows
# https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation#Generative_process
alpha = 0.1 # document_topic_prior
beta = 0.08 # topic_word_prior
K = 3 # number of topics
V = 9 # number of words
M = 10 # number of documents
N = 300 # number of words in the document, all documents have the same words
theta = dirichlet.rvs(alpha * np.ones(K), size=M)
# theta.shape = (M, K)
phi = dirichlet.rvs(beta * np.ones(V), size=K)
# phi.shape = (K, V)
w = np.zeros([M, N]) # word matrix, or corpus
z = np.zeros([M, N]) # topic assignment matrix
for i in range(M):
    for j in range(N):
        z[i][j] = np.random.choice(K, 1, p=theta[i, :])
        w[i][j] = np.random.choice(V, 1, p=phi[int(z[i][j]), :])

# observed data: w
# hidden variable: z, theta, phi
lda = LatentDirichletAllocation(n_components=K, doc_topic_prior=alpha,
    topic_word_prior=beta)
lda.fit(w)
print(lda.components_)
# estimation of theta
print(lda.transform(w))