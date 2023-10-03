import numpy as np


class OptimalSequentialGrouping(object):
    """
    Segmentation class which performs optimal sequential grouping (OSG)
    """

    def ismember(self, A, B):
        return np.asarray([np.sum(a == B) for a in A])

    def blockDivideDSum(self, D, K):
        """
        This method performs OSG using the sum objective function.
        :param D: input distance matrix
        :param K: number of blocks to divide the distance matrix into
        :return: the last index of each block
        """

        N = D.shape[0]

        if N < 1 or K < 1 or N != D.shape[1]:
            print("Error: Problem with input.")
            return []

        if K > N:
            print("Warning: More scenes than shots. Returning shot boundaries.")
            return np.arrange(1, N + 1)

        if K == 1:
            return [N - 1]

        D_sum = OptimalSequentialGrouping.calcDSum(self, D)

        C = np.zeros((N, K))
        I = np.zeros((N, K))

        # initialization
        for nn in range(0, N):
            C[nn, 0] = D_sum[nn, N - 1]
            I[nn, 0] = N - 1

        # fill the rest
        for kk in range(1, K):
            for nn in range(0, N - kk):
                # T will hold the vector in which we're searching for a minimum
                T = np.transpose(D_sum[nn, nn:N - kk]) + C[nn + 1:N - kk + 1, kk - 1]
                I[nn, kk] = np.argmin(T)
                C[nn, kk] = T[int(I[nn, kk])]
                I[nn, kk] = I[nn, kk] + nn

        # prepare returned boundaries
        boundary_locations = np.zeros(K)
        the_prev = -1
        for kk in range(0, K):
            boundary_locations[kk] = I[int(the_prev + 1), K - kk - 1]
            the_prev = boundary_locations[kk]

        if the_prev != N - 1:
            print("Warning: Encountered an unknown problem.")

        return boundary_locations

    def blockDivide2DSum(self, D1, D2, K, metric='average'):
        """
        This method performs multimodal OSG using the sum objective function.
        :param D1: input distance matrix 1
        :param D2: input distance matrix 2
        :param K: number of blocks to divide the distance matrix into
        :param metric: method to decide on joint outcome (default 'average', other options 'min', 'max')
        :return: the last index of each block
        """

        N = D1.shape[0]

        if N < 1 or K < 1 or N != D1.shape[1] or N != D2.shape[0] or N != D2.shape[1]:
            print("Error: Problem with input.")
            return []

        if K > N:
            print("Warning: More scenes than shots. Returning shot boundaries.")
            return np.arrange(1, N + 1)

        if K == 1:
            return [N - 1]

        D1_sum = OptimalSequentialGrouping.calcDSum(self, D1)
        D2_sum = OptimalSequentialGrouping.calcDSum(self, D2)

        C1 = np.zeros((N, K))
        C2 = np.zeros((N, K))
        I = np.zeros((N, K))

        # initialization
        for nn in range(0, N):
            C1[nn, 0] = D1_sum[nn, N - 1]
            C2[nn, 0] = D2_sum[nn, N - 1]
            I[nn, 0] = N - 1

        # fill the rest
        for kk in range(1, K):
            for nn in range(0, N - kk):
                # T will hold the vector in which we're searching for a minimum
                T1 = np.transpose(D1_sum[nn, nn:N - kk]) + C1[nn + 1:N - kk + 1, kk - 1]
                C1[nn, kk] = np.min(T1)
                T2 = np.transpose(D2_sum[nn, nn:N - kk]) + C2[nn + 1:N - kk + 1, kk - 1]
                C2[nn, kk] = np.min(T2)

                if T1.size < 2:
                    I[nn, kk] = nn
                else:
                    if metric == 'average':
                        E = (T1 - np.mean(T1)) / np.std(T1) + (T2 - np.mean(T2)) / np.std(T2)
                    elif metric == 'min':
                        E = np.minimum((T1 - np.mean(T1)) / np.std(T1), (T2 - np.mean(T2)) / np.std(T2))
                    elif metric == 'max':
                        E = np.maximum((T1 - np.mean(T1)) / np.std(T1), (T2 - np.mean(T2)) / np.std(T2))
                    else:
                        print('Error. Unrecognized metric: ' + metric + '. Performing average.')
                        E = (T1 - np.mean(T1)) / np.std(T1) + (T2 - np.mean(T2)) / np.std(T2)

                    I[nn, kk] = np.argmin(E) + nn

        # prepare returned boundaries
        boundary_locations = np.zeros(K)
        the_prev = -1
        for kk in range(0, K):
            boundary_locations[kk] = I[int(the_prev + 1), K - kk - 1]
            the_prev = boundary_locations[kk]

        if the_prev != N - 1:
            print("Warning: Encountered an unknown problem.")

        return boundary_locations

    def calcDSum(self, D):
        """
        This method calculates the intermediate sums of D.
        :param D: input distance matrix
        :return: D_sum. A matrix which the value at (i,j) is the sum of the values in D from ii to jj
        """

        N = D.shape[0]

        D_sum = np.zeros((N, N))

        for oo in range(1, N):
            for ii in range(0, N - oo):
                D_sum[ii, ii + oo] = 2 * D[ii, ii + oo] + D_sum[ii, ii + oo - 1] + D_sum[ii + 1, ii + oo] - D_sum[
                    ii + 1, ii + oo - 1]
                D_sum[ii + oo, ii] = D_sum[ii, ii + oo]

        return D_sum

    def DFromX(self, x, dist_type='euclidean'):
        if dist_type == 'euclidean':
            return np.linalg.norm(x[:, None] - x, axis=2, ord=2)
        elif dist_type == 'cosine':
            x_corr = np.matmul(x, x.T)
            x_square = np.diag(x_corr)
            x_square_rows = np.repeat(x_square[:, None], x.shape[0], axis=1)
            x_square_cols = x_square_rows.T
            return (1.0 - x_corr / (np.sqrt(x_square_rows * x_square_cols) + 1e-8)) / 2.0
        else:
            print("unrecognized distance type")
            return None

    def PR(self, input, ground_truth):
        TP = np.intersect1d(input,ground_truth).size
        return TP / input.size, TP / ground_truth.size

    def FCO(self, input, ground_truth):
        num_shots = int(ground_truth[-1])+1
        num_scenes = ground_truth.size

        gt_scene_numbering = np.zeros(num_shots)
        new_scene_numbering = np.zeros(num_shots)
        gt_scene_number = 0
        new_scene_number = 0
        for shot_ind in range(num_shots):
            gt_scene_numbering[shot_ind] = gt_scene_number
            new_scene_numbering[shot_ind] = new_scene_number
            if shot_ind == ground_truth[gt_scene_number]:
                gt_scene_number += 1
            if shot_ind == input[new_scene_number]:
                new_scene_number += 1

        Ct = np.zeros(num_scenes)
        C = 0
        Ot = np.zeros(num_scenes)
        O = 0

        for scene_num in range(num_scenes):
            (_, _, counts) = np.unique(new_scene_numbering[gt_scene_numbering == scene_num], return_index=True, return_counts=True)
            if counts.size == 0:
                freq = 0
            else:
                freq = np.max(counts)
            Ct[scene_num] = freq/np.sum(gt_scene_numbering == scene_num)
            C += Ct[scene_num] * np.sum(gt_scene_numbering == scene_num)
            Ot[scene_num] = sum(self.ismember(new_scene_numbering,np.unique(new_scene_numbering[gt_scene_numbering == scene_num])) & self.ismember(gt_scene_numbering, [scene_num-1, scene_num+1]))/np.sum(self.ismember(gt_scene_numbering, [scene_num-1, scene_num+1]))
            O += Ot[scene_num] * np.sum(gt_scene_numbering == scene_num)

        C = C/num_shots
        O = O/num_shots

        F = 2*C*(1-O)/(C+1-O)

        return F, C, O

    def estimate_num_scenes(self, input_D, D_t):
        if D_t is None:
            D = input_D / input_D.max()
        else:
            D = np.multiply(D_t, input_D / input_D.max())
        S = np.linalg.svd(D,compute_uv=False)
        the_graph = np.log(S[S>1])
        graph_length = len(the_graph)
        if graph_length < 2:
            return 1
        b = [graph_length - 1, the_graph[-1] - the_graph[0]]
        b_hat = b / np.sqrt(b[0]**2 + b[1]**2)
        p = np.transpose(np.vstack((np.arange(0,graph_length),the_graph-the_graph[0])))
        the_subt = p - np.transpose(np.vstack(((p*np.tile(b_hat, [graph_length, 1])).sum(1), (p*np.tile(b_hat, [graph_length, 1])).sum(1)))) * np.tile(b_hat,[graph_length, 1])
        the_dist = np.sqrt(the_subt[:, 0]**2 + the_subt[:, 1]**2)
        the_elbow = np.nonzero(the_dist==np.max(the_dist))[0][0]
        return the_elbow + 1


