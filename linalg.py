
import numpy as np

# using numpy arrays for matrices

def norm(v):
    # equivalent to:
    # acc = 0
    # for i in range(len(v)):
    #     acc += v[i]**2
    # return acc**(1/2)
    return np.sqrt(np.sum(np.square(v)))


def angle(v1, v2):
    # cos(alpha) = dot(v1,v2) / (norm(v1)*norm(v2))
    return np.arccos(dot(v1,v2) / (norm(v1) * norm(v2))


def orthogonal(v1, v2):
    return np.isclose(angle(v1, v2), (np.pi / 2))


def colinear(v1, v2):
    return np.isclose(angle(v1, v2), 0)


# this algorithm can suffer from catastrophic cancellation
# see Kahan summation algorithm
def dot(v1, v2):
    if len(v1) != len(v2):
        raise ValueError('Can only take dot-product of equal length vectors.')

    acc = 0
    for i in range(len(v1)):
        acc += v1[i] * v2[i]
    return acc


def scalar_projection(v_from, v_to):
    # equivalent to:
    # ||v_from|| cos(theta), with theta = acos(dot(v_from, v_to) / (norm(v_from) * norm(v_to)))
    # ||v_from|| dot(v_from, v_to) / (norm(v_from) * norm(v_to))
    # dot(v_from, v_to) / ||v_from||
    # TODO ?? can i bring the ||v_from|| inside the dot product?

    # b / norm(b) should be the unit vector in the direction of b
    return dot(v_from, (v_to / norm(v_to)))    


# so a matrix can *always* be converted into an upper triangular matrix through elementary row
# operations

# why are a set of vectors linearly independent iff the det of the matrix, w/ cols formed by the 
# vectors, is NOT zero?

# if a tuple (a1,...,an) with not ALL ai zero exists s.t. a1v1 + ... + anvn = 0, then
# the n vectors are linearly dependent. 
# (is the reverse true? special cases like one vi = 0 / scaled version of another?)


def indepenent(vectors, method='row_reduce'):
    if method == 'det':
        # forming matrix by taking vectors as columns
        # TODO does it matter?
        M = np.zeros(len(vectors[0]), len(vectors)))
        for i, v in enumerate(vectors):
            M[:,i] = v
        return det(M) == 0
        
    # TODO valid?
    elif method == 'row_reduce':
        pass

    else:
        raise ValueError('method can be det or row_reduce')
    

def multiply(A, B):
    if A.shape[1] != B.shape[0]:
        raise ValueError('To multiply A (mxn0) and B (n1xp), as AB, n0 must'+\
            ' equal n1. In general, AB != BA, i.e. matrix multiplication is not always commutative.')

    AB = np.zeros((A.shape[0], B.shape[1]))

    for i in range(A.shape[0]):
        for j in range(B.shape[1]):
            AB[i,j] = dot(A[i,:], B[:,j])
            # equivalent to:
            # for k in range(A.shape[1]):
            #     AB[i,j] += A[i,k] * B[k,j]

    return AB


def leading_coefficient_idx(v):
    """
    Doesn't check whether v is a zero vector.
    """
    return np.argmax(v.nonzero())


def in_row_echelon_form(A):
    """
    Returns true if in A:
        -all nonzero rows are above all zero rows
        -the leading coefficient (also called the pivot(?)) of a non-zero row
         is always strictly to the right of the row above it

    (implied that all entries beneath each leading coefficient must be zero) equivalent?

    Note: does not imply upper triangular nor implied by that, but they are equal in non-singular
    case (non-singular & some other condition, was it?)
    """

    zero_row = np.zeros(A.shape[1])
    saw_zero_row = False

    last_lead_index = 0

    for row in A:
        if np.isclose(row, zero_row):
            saw_zero_row = True

        # since we require all nonzero rows to be above all zero rows
        else if saw_zero_row:
            return False

        else:
            # should return index of first nonzero element
            idx = leading_coefficient_idx(row)
            if idx >= last_lead_index:
                return False
            last_lead_index = idx

    return True


def in_reduced_echelon_form(A):
    # TODO can only check by checking fixed point of elimination?
    pass


def gaussian_elimination(A):
    """
    Returns the row echelon form of A.

    Goal:
    -transform matrix through [row XOR col] operations to one that is in [row XOR col]
     echelon form

    TODO what / any additional operations needed to transform to *reduced* echelon form?
    added utility?

    Uses:
    -calculating determinants
    -"solving" systems of equations

    TODO always use augmented matrix? purpose if/if not?

    Operations available ("elementary row operations"):
    1: swap two rows
    2: multiply a row by a nonzero scalar
    3: add to one row a scalar multiple of another (seems to unecessarily include rule 2, 
       doesn't specifcy nonzero here either, on Wiki, but I think they mean to)

    -should run in O(n^3)
    TODO where n is number of elements or one dimension of (square?) matrix?
    -sometimes unstable, but generally not (for some classes of matrices at least)
    """
    # TODO can i just sort them so leading coefficient indices are <= row below and
    # proceed without resorting, or is it sometimes necessary to resort? maybe
    # depends on other choices?

    def enumerate_leads(M):
        return [(row_idx, leading_coefficient_idx(row)) for row_idx, row in enumerate(M)]

    # TODO test
    def sort_rows_by_leading_index(M):
        pairs = enumerate_leads(M)
        # sort output places keys from small to big
        sorted_row_indices = [x for x, y in sorted(pairs, key=lambda x: x[1])]
        return M[sorted_row_indices, :]

    A = sort_rows_by_leading_index(A)
    curr_row_idx = 0
    curr_lead = leading_coefficient_idx(A[curr_row_idx])

    # TODO stopping condition for reduced form?
    # this will return after reaching row echelon form
    while curr_row_idx < A.shape[0]:
        # find one row with same index for its leading coefficient
        # if none exist, continue to next row
        other_leads = enumerate_leads(A[curr_idx + 1:, :])
        # TODO assumes we were sorted going into this step (because idxs > than curr_lead ignored)
        # safe?
        ties = [i, l for i, l in other_leads if l == curr_lead]
        
        while len(ties) != 0:
            # TODO pops from end by default. OK?
            i, l = ties.pop()

            # TODO will i sometimes screw myself if i only subtract a multiple of the current row 
            # from the tied row? (rather than a row beneath)
            scale = l / A[curr_row_idx, curr_lead]
            A[i,:] = A[i,:] - scale * A[curr_row_idx, :]

        # TODO test. need to resort (and only / at least here)?
        A[curr_idx + 1, :] = sort_rows_by_leading_index(A[curr_idx + 1, :])
        # i guess this doesn't really need to be a while then?
        curr_row_idx += 1

    # TODO test
    return A


def rank(A):
    # TODO also accept a sequence of matrices and compute without multiplying
    # do this for other operations? can i do this in general for any?
    pass


def trace(A):
    """
    The sum of the diagonal of a square matrix.

    Properties:
        -invariant wrt change of basis (but not, it would seem, to elementary row operations??)
        -sum of (complex) eigenvalues
        -trace of product of matrices invariant to order and transposition for TWO matrices
         (really for all dimensions of X, Y? see Wiki)

         apparently NOT true generally, but true in some other special cases too.

        -linear as in: tr(A+B) = tr(A) + tr(B) and tr(cA) = c tr(A) (kinda obvious)

        for real matrices:
        -tr(X^TY) = sum_ij(XoY)_ij (the sum of the whole elementwise product?! TODO prove)
         where o = elementwise / Haddamard product (=> tr(X^TY) = vec(X)^T vec(Y))

        "If A is a linear operator represented by a square n-by-n matrix with real or complex 
	 entries and if λ1, ..., λn are the eigenvalues of A... then

	 tr(A) = \sum_i{\lambda_i}

	 This follow from the fact that A is always similar to its Jordan form, an upper triangular
	 matrix having λ1, ..., λn on the main diagonal. In contrast, the determinant of A is the
	 product of its eigenvalues; i.e.,

	 det(A) = \prod_i{\lambda_i}"

	 -

	 -is this dependent on some detail of how they define linear operator? or is the sum of the
	  eigenvalues always there in the sum of the diagonal?

	 Corresponds to the derivative of the determinant!! (see Jacobi's formula)
    """
    # TODO should i define it for some non-square matrices?
    if A.shape[0] != A.shape[1]:
        raise ValueError('trace not defined for non-square matrices')
    acc = 0
    for i in range(A.shape[0]):
        acc += A[i,i]
    return acc


def det(A, method='elimination'):
    if method == 'elimination':

    elif method == 'elementary':

    else:
        raise ValueError('method not valid')


def characteristic(A):
    """
    Returns sympy expression for characteristic expression of A.

    -c(A) = c(B) necessary for A similar to B, but not sufficient
     where "similar to" = there exists P s.t. A = P^-1BP

    """
    pass


def are_similar(A, B):
    """
    ONLY DEFINED FOR SQUARE MATRICES (this is key distinction from matrix "equivalence")
    similar => equivalent, but not other way

    paraphrasing SO answer:
    -will probably need to use Jordan / Frobenius ("rational") normal forms
     (up to order or diagonal elements?)

     comment: "...Jordan decomp cannot be stably computed...[use] Schur decomposition"
     another commend: "[above comment] implies that w/ finite precision, similarity of matrices
     cannot be decided, period"

     ...is approximate similarity useful?

    -not enough to compare characteristic polynomial / reduced row echelon form
    -"The reduced row echelon form and the reduced column echelon form will not help you,
      because any two invertible matrices have the same forms (the identity), but need not have
      the same determinant (so they will not be similar)."
    """
    pass


def are_equivalent(A, B):
    """
    For some invertible nxn P and invertible mxm Q, B = Q^-1 AP.

    Means A and B represent "same" linear transform, under different choice of bases V and W,
    with P and Q being the c.o.b matrices in V and W.

    equivalent => can be transformed into each other by elementary row and column operations.
    equivalent => share row space
    """
    pass


def are_congruent(A, B):
    """
    There exists an invertible P "over same field" s.t. P^TAP = B

    somehow related to "bilinear forms", "quadratic forms", and their Gram matrices

    -symmetric, real matrices that are congruent have same # of +, - and zero e.v.s
    """
    pass
