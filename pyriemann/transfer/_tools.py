import numpy as np
from numpy.random import Generator, PCG64
import scipy
from copy import deepcopy
from pyriemann.estimation import ERPCovariances
from pyriemann.utils.tangentspace import transport, tangent_space

def encode_domains(X, y, domain):
    r"""Encode the domains of the matrices in the labels.

    We handle the possibility of having different domains for the datasets by
    extending the labels of the matrices and including this information to
    them. For instance, if we have a matrix X with class `left_hand` on the
    `domain_01` then its extended label will be `domain_01/left_hand`. Note
    that if the classes were integers at first, they will be converted to
    strings.

    Parameters
    ----------
    X : ndarray, shape (n_matrices, n_channels, n_channels)
        Set of SPD matrices.
    y : ndarray, shape (n_matrices,)
        Labels for each matrix.
    domain : ndarray, shape (n_matrices,)
        Domains for each matrix.

    Returns
    -------
    X_enc : ndarray, shape (n_matrices, n_channels, n_channels)
        The same set of SPD matrices given as input.
    y_enc : ndarray, shape (n_matrices,)
        Extended labels for each matrix.

    See Also
    --------
    decode_domains

    Notes
    -----
    .. versionadded:: 0.4
    """
    if len(y) != len(domain):
        raise ValueError("Input lengths don't match")

    y_enc = [str(d_) + '/' + str(y_) for (d_, y_) in zip(domain, y)]
    return X, np.array(y_enc)


def decode_domains(X_enc, y_enc):
    """Decode the domains of the matrices in the labels.

    We handle the possibility of having different domains for the datasets by
    encoding the domain information into the labels of the matrices. This
    method converts the data into its original form, with a separate data
    structure for labels and for domains.

    Parameters
    ----------
    X_enc : ndarray, shape (n_matrices, n_channels, n_channels)
        Set of SPD matrices.
    y_enc : ndarray, shape (n_matrices,)
        Extended labels for each matrix.

    Returns
    -------
    X : ndarray, shape (n_matrices, n_channels, n_channels)
        Set of SPD matrices.
    y : ndarray, shape (n_matrices,)
        Labels for each matrix.
    domain : ndarray, shape (n_matrices,)
        Domains for each matrix.

    See Also
    --------
    encode_domains

    Notes
    -----
    .. versionadded:: 0.4
    """
    y, domain = [], []
    for y_enc_ in y_enc:
        y_dec_ = y_enc_.split('/')
        domain.append(y_dec_[-2])
        y.append(y_dec_[-1])
    return X_enc, np.array(y), np.array(domain)


class TLSplitter():
    """Class for handling the cross-validation splits of multi-domain data.

    This is a wrapper to sklearn's cross-validation iterators [1]_ which
    ensures the handling of domain information with the data points. In fact,
    the data from source domain is always fully available in the training
    partition whereas the random splits are done on the data points from the
    target domain.

    Parameters
    ----------
    target_domain : str
        Domain considered as target.
    cv : None | BaseCrossValidator | BaseShuffleSplit, default=None
        An instance of a cross validation iterator from sklearn.

    References
    ----------
    .. [1] https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators

    Notes
    -----
    .. versionadded:: 0.4
    """  # noqa
    def __init__(self, target_domain, cv):

        self.target_domain = target_domain
        self.cv = cv

    def split(self, X, y):
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD matrices.
        y : ndarray, shape (n_matrices,)
            Extended labels for each matrix.

        Yields
        ------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        """

        # decode the domains of the data points
        X, y, domain = decode_domains(X, y)

        # indentify the indices of the target dataset
        idx_source = np.where(domain != self.target_domain)[0]
        idx_target = np.where(domain == self.target_domain)[0]
        y_target = y[idx_target]

        # index of training-split for the target data points
        ss_target = self.cv.split(idx_target, y_target)
        for train_sub_idx_target, test_sub_idx_target in ss_target:
            train_idx = np.concatenate(
                [idx_source, idx_target[train_sub_idx_target]])
            test_idx = idx_target[test_sub_idx_target]
            yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None):
        """Returns the number of splitting iterations in the cross-validator.

        Parameters
        ----------
        X : object
            Ignored, exists for compatibility.
        y : object
            Ignored, exists for compatibility.

        Returns
        -------
        n_splits : int
            Returns the number of splitting iterations in the cross-validator.
        """
        return self.cv.n_splits

class GLUniteDomains():

    def __init__(self, cv):
        self.cv = cv
        self.train_vecs = []
        self.test_vecs = []
        self.train_labels = []
        self.test_labels = []
    
    def split(self, X, y):
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD matrices.
        y : ndarray, shape (n_matrices,)
            Extended labels for each matrix.

        Yields
        ------
        X_train : ndarray
            The training set for that split.
        X_test : ndarray
            The testing set for that split.
        y_train : ndarray
            The training set labels.
        y_test : ndarray
            The testing set labels.
        """
        temp_train = []
        temp_test = []
        temp_train_label = []
        temp_test_label = []
        for nsp, (train_idx, test_idx) in enumerate(self.cv.split(X, y)):
            # Train/test EEG data and label information
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Create prototype of Target class using ONLY the train data
            # Transform test data using the same prototype
            prt_estimator = ERPCovariances(classes=["Target"], estimator='lwf')
            covs_train = prt_estimator.fit_transform(X_train, y_train)
            covs_test = prt_estimator.transform(X_test)

            # Recentering around Identity matrix
            I = np.eye(covs_train[0].shape[0])
            covs_train = transport(covs_train, I, metric="riemann")
            covs_test = transport(covs_test, I, metric="riemann")

            # Tangent space mapping
            train_tsvecs = tangent_space(covs_train, I, metric="riemann")
            test_tsvecs = tangent_space(covs_test, I, metric="riemann")

            # Append each split into a list
            temp_train.append(train_tsvecs.T)
            temp_test.append(test_tsvecs.T)

            temp_train_label.append(y_train)
            temp_test_label.append(y_test)
        
        # Append each subject into a list
        self.train_vecs.append(temp_train)
        self.test_vecs.append(temp_test)

        self.train_labels.append(temp_train_label)
        self.test_labels.append(temp_test_label)



def create_bootstrap(features, labels, bootsize = 25, n_of_boots = 100, w = "u", random_state = 1):
    # Create pseudo-random seeds for random integer generator
    seeds = Generator(PCG64(random_state)).integers(1,1e12, n_of_boots)
    classes = np.unique(labels)
    boot_all = []
    for c in classes:
        if w == "u":
            weights = np.ones(bootsize) / bootsize
        else:
            weights = Generator(PCG64(random_state)).random(bootsize)
            weights = weights / np.sum(weights)
        
        temp_boot = []
        temp = features[:,labels==c]
        [temp_boot.append(temp[:,Generator(PCG64(seeds[x])).integers(1,temp.shape[1],bootsize)].dot(weights)) for x in range(n_of_boots)]
        boot_all.append(np.array(temp_boot))
        
    return np.transpose(np.concatenate(boot_all))

def whiten_data(bootstraps, wh_type="smart", white_dim=16, smart_subspace=16, verbose=True):
    # Initialize whitening matrices
    M = len(bootstraps)
    D = white_dim
    L = bootstraps[0].shape[1]
    
    boldT = [np.zeros((D,L)) for m in range(M)]
    boldS = [None for m in range(M)]
    boldW = [np.zeros((D,D)) for m in range(M)]

    # Whitening begins
    verbose and print("Pre-whitening started...")
    if wh_type == "svd":
        for m in range(M):
            # order of svd return >>> U, S, VH
            boldS[m] = scipy.linalg.svd(bootstraps[m])
            boldW[m] = boldS[m][0][:,0:white_dim] * np.transpose(1/boldS[m][1][0:white_dim])
            boldT[m] = boldW[m].T @ bootstraps[m]
            verbose and print("Completed..... %" + str(100*m/M))
        verbose and print("Pre-whitening is completed with >>>  " + wh_type)
    elif wh_type == "smart":
        temp_T = [None for m in range(M)]
        boldC = [None for m in range(M)]
        for m in range(M):
            boldS[m] = scipy.linalg.svd(bootstraps[m])
            temp_T[m] = (boldS[m][0][:,0:white_dim] @ np.diag(boldS[m][1][0:white_dim])).T @ bootstraps[m]
            boldC[m] = [np.zeros((D,D)) for m in range(M)]
        verbose and print("Pre-whitening step-1 is completed...")

        for m in range(M):
            for i in range(m+1,M):
                boldC[i][m] = temp_T[i] @ temp_T[m].T
        verbose and print("Pre-whitening step-2 is completed...")

        H = np.zeros((D,D))
        for m in range(M):
            H = np.zeros((D,D))
            for j in range(M):
                if m>j:
                    H += boldC[m][j]
                if m<j:
                    H += boldC[j][m].T
            F = scipy.linalg.svd(H)
            boldW[m] = F[0][:,0:smart_subspace]
        verbose and print("Pre-whitening matrices are estimated...")

        for m in range(M):
            boldT[m] = boldW[m].T @ temp_T[m]
        verbose and print("Pre-whitening is completed with >>>  " + wh_type)
    else:
        boldT = bootstraps
        verbose and print("No pre-whitening is applied to bootstraps!")
    return boldT, boldS, boldW

def norm_U(boldU, norm_type = "unit", boldT = []):
    
    M = len(boldU)
    if norm_type == "unit":
        new_boldU = [boldU[m] / scipy.linalg.norm(boldU[m], axis=0) for m in range(M)]
        
    elif norm_type == "white":
        if not boldT:
            raise ValueError("boldT cannot be empty for smart whitening!!!")
        if len(boldU) != len(boldT):
            raise ValueError("boldU and boldT list should have same length!!!")
        new_boldU = []
        for m in range(M):
            tempU = boldU[m]
            P = boldT[m] @ boldT[m].T
            bk = np.array([np.sqrt(tempU[:,k].T @ P @ tempU[:,k]) for k in range(tempU.shape[0])])
            new_boldU.append(tempU / bk.T)
    else:
        print("boldU matrices are not normalized and returned as they are.")
        print("white or unit norm type can be chosen for normalization.")

    return new_boldU

def estimate_B(boldU, boldW, wh_type = "smart", white_dim = 16, reverse_selection = False, boldS = []):
    if len(boldU) != len(boldW):
        raise ValueError("Number of alignment matrices is not equal to number of whitening matrices!!!")
    if (wh_type == "smart") & (not boldS):
        raise ValueError("boldS can not be empty for smart whitening!!!")
    
    M = len(boldU)
    B = []

    if wh_type == "svd":
        B = [boldW[m] @ boldU[m] for m in range(M)]
    elif wh_type == "smart":
        B = [boldS[m][0][:,0:white_dim] @ np.diag(boldS[m][1][0:white_dim]) @ boldW[m] @ boldU[m] for m in range(M)]
    else:
        print("The chosen whitening type is not proper. No B matrices estimated.")
        print("Make sure the whitening type is either \"smart\" or \"svd\".")
    return B

def align_features(train_split, test_split, B, sub_dim = None):
    if sub_dim is None:
        sub_dim = B[0].shape[1]
    
    aligned_train = [B[m][:,0:sub_dim].T @ f for m, f in enumerate(train_split)]
    aligned_test = [B[m][:,0:sub_dim].T @ f for m, f in enumerate(test_split)]

    return aligned_train, aligned_test

# shape of the feature matrix should be in the order of
# features X trials

def zero_mean_vecs(vecs):
    return np.transpose(deepcopy(vecs.T) - np.mean(vecs, axis=1))

def norm_vecs(vecs):
    return deepcopy(vecs) / np.mean(scipy.linalg.norm(vecs, axis=0))

def cross_cov(X):
    M = len(X)
    C = [[[] for i in range(M)] for j in range(M)]
    for i in range(M):
        for j in range(i,M):
            C[i][j] = X[i] @ X[j].T / X[i].shape[1]
            C[j][i] = C[i][j].T
    return C

def flipcol(boldU, M, n, e):
    U = deepcopy(boldU)
    for i in range(M):
        temp = deepcopy(U[i][:,e])
        U[i][:,e] = deepcopy(U[i][:,n])
        U[i][:,n] = deepcopy(temp)
    return U

def flip_and_permute(boldU, boldX, M, k, input_="d", covest="SCM", dims=1, meanX=0, trace=False):
    # if the input_ is "d", SCM cross-covariances are estimated
    if input_ == "d":
        boldC = cross_cov(boldX)
    else:
        boldC = boldX
    
    U = deepcopy(boldU)
    udim = U[0].shape[0]
    D = [[] for i in range(M)]
    for i in range(M):
        for j in range(M):
            D[i].append(np.diag([U[i][:,n].T @ boldC[i][j] @ U[j][:,n] for n in range(udim)]))
            
    for e in range(udim):
        maxx = 0.
        # find the position of the absolute maximum
        for i in range(M-1):
            for j in range(i+1,M):
                for n in range(e,udim):
                    absd = abs(D[i][j][n,n])
                    if absd > maxx:
                        maxx = deepcopy(absd)
                        p = deepcopy([i, j, n])
        
        # flip sign of boldU[j][n,n] if absolute maximum is negative
        i, j, n = p[0], p[1], p[2]
        if D[i][j][n,n] < 0:
            U[j][:,n] *= -1
        
        # flip sign of boldU[j] for all j≠i, i=1:M if their corresponding element is negative
        for x in range(M):
            if x != j:
                if D[i][x][n,n] < 0:
                    U[x][:,n] *= -1.
        
        # bring the maximum from position n to on top (for current e)
        if n != e:
            U = flipcol(U, M, n, e)
        
        # compute D again
        D = [[] for i in range(M)]
        for j in range(M):
            for i in range(M):
                D[i].append(np.diag([U[i][:,n].T @ boldC[i][j] @ U[j][:,n] for n in range(udim)]))

    return U
