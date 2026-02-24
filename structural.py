import torch

def add_feats(x_t, edge_attr_t, t, mask, mol):
    device = x_t.device
    # if mol:
    #
    #     valencies = torch.tensor([4, 3, 2, 1, 3, 1, 1, 1, 5, 2, 2, 4]).to(device)
    #     weights = torch.tensor([12.01, 14.01, 16.00, 19.00, 10.81, 79.90, 35.45, 126.90, 30.97, 32.07, 78.96, 28.09]).to(device)
    #
    #     atoms = torch.argmax(x_t, dim=-1)
    #     atom_val = valencies[atoms].unsqueeze(-1)
    #     atom_weight = weights[atoms].unsqueeze(-1)
    #     mol_weight = atom_weight.reshape(-1, 9).sum(dim=1).unsqueeze(-1).repeat(1, 9).reshape(-1, 1)
    #     mol_weight = mol_weight.reshape(x_t.shape[0], 9, 1)
    #
    #     x_t = torch.cat([x_t, atom_val, atom_weight, mol_weight], dim=-1)

    # t = torch.tensor([t]).to(device)
    y_t = t.squeeze().unsqueeze(-1).to(device)
    #
    # graph_t = graph_1.clone()
    # graph_t.x, graph_t.edge_attr = x_t, edge_attr_t
    #
    # graphs = Batch.to_data_list(graph_t)
    #
    # adj = [torch_geometric.utils.to_dense_adj(graph.edge_index, edge_attr=torch.argmax(graph.edge_attr, dim=-1),
    #                                           max_num_nodes=9) for graph in graphs]
    # # stack
    #
    #
    #
    # sizes = [graph.nodes_wo_mask.item() for graph in graphs]
    # A = torch.stack(adj).squeeze()
    # # set diagonals to zero
    # A = A * (~torch.eye(A.shape[-1]).bool()).unsqueeze(0).to(device)
    # # one hot
    # A = torch.nn.functional.one_hot(A.long(), num_classes=4).float()

    A = edge_attr_t[..., 1:].sum(dim=-1).float()


    mask = mask.reshape(-1, A.shape[1])

    A = edge_attr_t[..., 1:].sum(dim=-1).float() * mask.unsqueeze(1) * mask.unsqueeze(2)
    L = compute_laplacian(A, normalize=False)
    mask_diag = 2 * L.shape[-1] * torch.eye(A.shape[-1]).type_as(L).unsqueeze(0)
    mask_diag = mask_diag * (~mask.unsqueeze(1)) * (~mask.unsqueeze(2))
    L = L * mask.unsqueeze(1) * mask.unsqueeze(2) + mask_diag

    eigvals, eigvectors = torch.linalg.eigh(L)
    eigenvalues = eigvals.type_as(A) / torch.sum(mask, dim=1, keepdim=True)
    eigvectors = eigvectors * mask.unsqueeze(2) * mask.unsqueeze(1)

    y_a, y_b = get_eigenvalues_features(eigenvalues, k=5)
    not_lcc_indicator, first_k_ev = get_eigenvectors_features(eigvectors, mask, y_a, k=2)

    # not_lcc_indicator = not_lcc_indicator.reshape(not_lcc_indicator.shape[0] * not_lcc_indicator.shape[1], 1)
    # first_k_ev = first_k_ev.reshape(first_k_ev.shape[0] * first_k_ev.shape[1], 2)

    y_t = torch.cat([y_t, y_a, y_b], dim=-1)
    x_t = torch.cat([x_t, not_lcc_indicator, first_k_ev], dim=-1)

    return x_t, y_t

    assert (A >= -0.1).all()

    k1_matrix = A.float()
    d = A.sum(dim=-1)
    k2_matrix = k1_matrix @ A.float()
    k3_matrix = k2_matrix @ A.float()
    k4_matrix = k3_matrix @ A.float()
    k5_matrix = k4_matrix @ A.float()
    k6_matrix = k5_matrix @ A.float()

    assert (k1_matrix >= -0.1).all()
    assert (k2_matrix >= -0.1).all()
    assert (k3_matrix >= -0.1).all()
    assert (k4_matrix >= -0.1).all()
    assert (k5_matrix >= -0.1).all()
    assert (k6_matrix >= -0.1).all()

    c3 = batch_diagonal(k3_matrix) / 2
    k3x, k3y = (c3 / 2).unsqueeze(-1).float(), (torch.sum(c3, dim=-1) / 6).unsqueeze(-1).float()

    diag_a4 = batch_diagonal(k4_matrix)
    print(d)

    c4 = diag_a4 - d * (d - 1) - (A @ d.unsqueeze(-1)).sum(dim=-1)
    k4x, k4y = (c4 / 2).unsqueeze(-1).float(), (torch.sum(c4, dim=-1) / 8).unsqueeze(-1).float()

    diag_a5 = batch_diagonal(k5_matrix).float()
    triangles = batch_diagonal(k3_matrix).float()

    c5 = diag_a5 - 2 * triangles * d - (A @ triangles.unsqueeze(-1)).sum(dim=-1) + triangles
    k5x, k5y = (c5 / 2).unsqueeze(-1).float(), (c5.sum(dim=-1) / 10).unsqueeze(-1).float()

    term_1_t = batch_trace(k6_matrix)
    term_2_t = batch_trace(k3_matrix ** 2)
    term3_t = torch.sum(A * k2_matrix.pow(2), dim=[-2, -1])
    d_t4 = batch_diagonal(k2_matrix)
    a_4_t = batch_diagonal(k4_matrix)
    term_4_t = (d_t4 * a_4_t).sum(dim=-1)
    term_5_t = batch_trace(k4_matrix)
    term_6_t = batch_trace(k3_matrix)
    term_7_t = batch_diagonal(k2_matrix).pow(3).sum(-1)
    term8_t = torch.sum(k3_matrix, dim=[-2, -1])
    term9_t = batch_diagonal(k2_matrix).pow(2).sum(-1)
    term10_t = batch_trace(k2_matrix)

    c6_t = (term_1_t - 3 * term_2_t + 9 * term3_t - 6 * term_4_t + 6 * term_5_t - 4 * term_6_t + 4 * term_7_t +
            3 * term8_t - 12 * term9_t + 4 * term10_t)

    _, k6y = None, (c6_t / 12).unsqueeze(-1).float()

    assert (k3x >= -0.1).all()
    assert (k4x >= -0.1).all()
    assert (k5x >= -0.1).all()
    assert (k6y >= -0.1).all()

    kcyclesx = torch.cat([k3x, k4x, k5x], dim=-1)
    kcyclesy = torch.cat([k3y, k4y, k5y, k6y], dim=-1)

    kcyclesx = kcyclesx.reshape(kcyclesx.shape[0] * kcyclesx.shape[1], 3)

    x_t = torch.cat([x_t, kcyclesx], dim=-1)
    y_t = torch.cat([y_t, kcyclesy], dim=-1)

    return x_t, y_t




    print(c3, c3.shape)
    print(k5_matrix)
    quit()

    return x_t, y_t
    print(y_t.shape, x_t.shape)


    quit()


    # get eigenvalues and eigenvectors
    E_t = noisy_data['E_t']
    mask = noisy_data['node_mask']
    A = E_t[..., 1:].sum(dim=-1).float() * mask.unsqueeze(1) * mask.unsqueeze(2)
    L = compute_laplacian(A, normalize=False)
    mask_diag = 2 * L.shape[-1] * torch.eye(A.shape[-1]).type_as(L).unsqueeze(0)
    mask_diag = mask_diag * (~mask.unsqueeze(1)) * (~mask.unsqueeze(2))
    L = L * mask.unsqueeze(1) * mask.unsqueeze(2) + mask_diag
    eigvals = torch.linalg.eigvalsh(L)  # bs, n
    eigvals = eigvals.type_as(A) / torch.sum(mask, dim=1, keepdim=True)

    n_connected_comp, batch_eigenvalues = get_eigenvalues_features(eigenvalues=eigvals)
    return n_connected_comp.type_as(A), batch_eigenvalues.type_as(A)

    num_zeros, non_zeros, non_zero_vecs = [], [], []

    # for adj_mat, size in zip(adj, sizes):
        # do eigen decomposition
        # a = adj_mat[:size, :size].float()
        # deg = torch.diag(torch.sum(a, dim=-1).pow(-0.5))
        # # replace inf with 0
        # deg[deg == float('inf')] = 0
        # lap = torch.eye(size).to(device) - torch.matmul(torch.matmul(deg, a), deg)
        #
        # L, Q = torch.linalg.eigh(lap.float())
        #
        #
        # # number of 0 eigenvalues
        # num_zero = torch.sum(torch.abs(L) < 1e-5)
        # # first 5 non zero eigenvalues
        # non_zero = torch.zeros(5)
        # evs = torch.sort(torch.abs(L))[0][num_zero: num_zero + 5]
        # non_zero[:len(evs)] = evs
        #
        # # get eigenvectors for first 2 eigenvectors non zero
        # non_zero_vec = torch.zeros(size=(9, 8))
        # non_zero_vz = Q[:, num_zero: num_zero + 8]
        # non_zero_vec[:non_zero_vz.shape[0], :non_zero_vz.shape[1]] = non_zero_vz
        #
        # num_zeros.append(num_zero)
        # non_zeros.append(non_zero)
        # non_zero_vecs.append(non_zero_vec)

    # make into vectors

    print('A')
    with Pool(processes=8) as pool:
        results = pool.map(add_pe_graph, list(zip(adj, sizes)))

    print('B')
    num_zeros, non_zeros, non_zero_vecs = zip(*results)

    num_zeros = torch.tensor(num_zeros).to(device)
    non_zeros = torch.stack(non_zeros).to(device)
    non_zero_vecs = torch.stack(non_zero_vecs).to(device)

    y_t = torch.cat([y_t, num_zeros.unsqueeze(-1), non_zeros], dim=-1)

    batch_size = non_zero_vecs.shape[0]
    nodes = 9

    non_zero_vecs = non_zero_vecs.reshape(batch_size * nodes, 8)

    x_t = torch.cat([x_t, non_zero_vecs], dim=-1)

    return x_t, y_t


def add_pe_graph(graph_data):
    adj_mat, size = graph_data
    a = adj_mat[:size, :size].float()
    deg = torch.diag(torch.sum(a, dim=-1).pow(-0.5))
    # replace inf with 0
    deg[deg == float('inf')] = 0
    lap = torch.eye(size).to(adj_mat.device) - torch.matmul(torch.matmul(deg, a), deg)

    L, Q = torch.linalg.eigh(lap.float())

    # number of 0 eigenvalues
    num_zero = torch.sum(torch.abs(L) < 1e-5)
    # first 5 non zero eigenvalues
    non_zero = torch.zeros(5)
    evs = torch.sort(torch.abs(L))[0][num_zero: num_zero + 5]
    non_zero[:len(evs)] = evs

    # get eigenvectors for first 2 eigenvectors non zero
    non_zero_vec = torch.zeros(size=(9, 8))
    non_zero_vz = Q[:, num_zero: num_zero + 8]
    non_zero_vec[:non_zero_vz.shape[0], :non_zero_vz.shape[1]] = non_zero_vz

    return num_zero, non_zero, non_zero_vec



def compute_laplacian(adjacency, normalize: bool):
    """
    adjacency : batched adjacency matrix (bs, n, n)
    normalize: can be None, 'sym' or 'rw' for the combinatorial, symmetric normalized or random walk Laplacians
    Return:
        L (n x n ndarray): combinatorial or symmetric normalized Laplacian.
    """
    diag = torch.sum(adjacency, dim=-1)     # (bs, n)
    n = diag.shape[-1]
    D = torch.diag_embed(diag)      # Degree matrix      # (bs, n, n)
    combinatorial = D - adjacency                        # (bs, n, n)

    if not normalize:
        return (combinatorial + combinatorial.transpose(1, 2)) / 2

    diag0 = diag.clone()
    diag[diag == 0] = 1e-12

    diag_norm = 1 / torch.sqrt(diag)            # (bs, n)
    D_norm = torch.diag_embed(diag_norm)        # (bs, n, n)
    L = torch.eye(n).unsqueeze(0) - D_norm @ adjacency @ D_norm
    L[diag0 == 0] = 0
    return (L + L.transpose(1, 2)) / 2

def get_eigenvalues_features(eigenvalues, k=5):
    """
    values : eigenvalues -- (bs, n)
    node_mask: (bs, n)
    k: num of non zero eigenvalues to keep
    """
    ev = eigenvalues
    bs, n = ev.shape
    n_connected_components = (ev < 1e-5).sum(dim=-1)
    assert (n_connected_components > 0).all(), (n_connected_components, ev)

    to_extend = max(n_connected_components) + k - n
    if to_extend > 0:
        eigenvalues = torch.hstack((eigenvalues, 2 * torch.ones(bs, to_extend).type_as(eigenvalues)))
    indices = torch.arange(k).type_as(eigenvalues).long().unsqueeze(0) + n_connected_components.unsqueeze(1)
    first_k_ev = torch.gather(eigenvalues, dim=1, index=indices)
    return n_connected_components.unsqueeze(-1), first_k_ev


def get_eigenvectors_features(vectors, node_mask, n_connected, k=2):
    """
    vectors (bs, n, n) : eigenvectors of Laplacian IN COLUMNS
    returns:
        not_lcc_indicator : indicator vectors of largest connected component (lcc) for each graph  -- (bs, n, 1)
        k_lowest_eigvec : k first eigenvectors for the largest connected component   -- (bs, n, k)
    """
    bs, n = vectors.size(0), vectors.size(1)

    # Create an indicator for the nodes outside the largest connected components
    first_ev = torch.round(vectors[:, :, 0], decimals=3) * node_mask                        # bs, n
    # Add random value to the mask to prevent 0 from becoming the mode
    random = torch.randn(bs, n, device=node_mask.device) * (~node_mask)                                   # bs, n
    first_ev = first_ev + random
    most_common = torch.mode(first_ev, dim=1).values                                    # values: bs -- indices: bs
    mask = ~ (first_ev == most_common.unsqueeze(1))
    not_lcc_indicator = (mask * node_mask).unsqueeze(-1).float()

    # Get the eigenvectors corresponding to the first nonzero eigenvalues
    to_extend = max(n_connected) + k - n
    if to_extend > 0:
        vectors = torch.cat((vectors, torch.zeros(bs, n, to_extend).type_as(vectors)), dim=2)   # bs, n , n + to_extend
    indices = torch.arange(k).type_as(vectors).long().unsqueeze(0).unsqueeze(0) + n_connected.unsqueeze(2)    # bs, 1, k
    indices = indices.expand(-1, n, -1)                                               # bs, n, k
    first_k_ev = torch.gather(vectors, dim=2, index=indices)       # bs, n, k
    first_k_ev = first_k_ev * node_mask.unsqueeze(2)

    return not_lcc_indicator, first_k_ev



def batch_trace(X):
    """
    Expect a matrix of shape B N N, returns the trace in shape B
    :param X:
    :return:
    """
    diag = torch.diagonal(X, dim1=-2, dim2=-1)
    trace = diag.sum(dim=-1)
    return trace


def batch_diagonal(X):
    """
    Extracts the diagonal from the last two dims of a tensor
    :param X:
    :return:
    """
    return torch.diagonal(X, dim1=-2, dim2=-1)


class KNodeCycles:
    """ Builds cycle counts for each node in a graph.
    """

    def __init__(self):
        super().__init__()

    def calculate_kpowers(self):
        self.k1_matrix = self.adj_matrix.float()
        self.d = self.adj_matrix.sum(dim=-1)
        self.k2_matrix = self.k1_matrix @ self.adj_matrix.float()
        self.k3_matrix = self.k2_matrix @ self.adj_matrix.float()
        self.k4_matrix = self.k3_matrix @ self.adj_matrix.float()
        self.k5_matrix = self.k4_matrix @ self.adj_matrix.float()
        # self.k6_matrix = self.k5_matrix @ self.adj_matrix.float()

    def k3_cycle(self):
        """ tr(A ** 3). """
        c3 = batch_diagonal(self.k3_matrix)
        return (c3 / 2).unsqueeze(-1).float(), (torch.sum(c3, dim=-1) / 6).unsqueeze(-1).float()

    def k4_cycle(self):
        diag_a4 = batch_diagonal(self.k4_matrix)
        c4 = diag_a4 - self.d * (self.d - 1) - (self.adj_matrix @ self.d.unsqueeze(-1)).sum(dim=-1)
        return (c4 / 2).unsqueeze(-1).float(), (torch.sum(c4, dim=-1) / 8).unsqueeze(-1).float()

    def k5_cycle(self):
        diag_a5 = batch_diagonal(self.k5_matrix)
        triangles = batch_diagonal(self.k3_matrix)
        c5 = diag_a5 - 2 * triangles * self.d - (self.adj_matrix @ triangles.unsqueeze(-1)).sum(dim=-1) + triangles
        return (c5 / 2).unsqueeze(-1).float(), (c5.sum(dim=-1) / 10).unsqueeze(-1).float()

    def k6_cycle(self):
        term_1_t = batch_trace(self.k6_matrix)
        term_2_t = batch_trace(self.k3_matrix ** 2)
        term3_t = torch.sum(self.adj_matrix * self.k2_matrix.pow(2), dim=[-2, -1])
        d_t4 = batch_diagonal(self.k2_matrix)
        a_4_t = batch_diagonal(self.k4_matrix)
        term_4_t = (d_t4 * a_4_t).sum(dim=-1)
        term_5_t = batch_trace(self.k4_matrix)
        term_6_t = batch_trace(self.k3_matrix)
        term_7_t = batch_diagonal(self.k2_matrix).pow(3).sum(-1)
        term8_t = torch.sum(self.k3_matrix, dim=[-2, -1])
        term9_t = batch_diagonal(self.k2_matrix).pow(2).sum(-1)
        term10_t = batch_trace(self.k2_matrix)

        c6_t = (term_1_t - 3 * term_2_t + 9 * term3_t - 6 * term_4_t + 6 * term_5_t - 4 * term_6_t + 4 * term_7_t +
                3 * term8_t - 12 * term9_t + 4 * term10_t)
        return None, (c6_t / 12).unsqueeze(-1).float()

    def k_cycles(self, adj_matrix, verbose=False):
        self.adj_matrix = adj_matrix
        self.calculate_kpowers()

        k3x, k3y = self.k3_cycle()
        assert (k3x >= -0.1).all()

        k4x, k4y = self.k4_cycle()
        assert (k4x >= -0.1).all()

        k5x, k5y = self.k5_cycle()
        assert (k5x >= -0.1).all(), k5x

        # _, k6y = self.k6_cycle()
        # assert (k6y >= -0.1).all()

        kcyclesx = torch.cat([k3x, k4x, k5x], dim=-1)
        kcyclesy = torch.cat([k3y, k4y, k5y], dim=-1)
        return kcyclesx, kcyclesy


def generate_random_categorical(num_graphs, num_nodes, mask):
    x_sample = torch.randint(0, 4, size=(num_graphs, num_nodes))
    e_sample = torch.randint(0, 4, size=(num_graphs, num_nodes, num_nodes))

    float_mask = mask.float()
    x_sample = x_sample.type_as(float_mask)
    e_sample = e_sample.type_as(float_mask)

    # Get upper triangular part of edge noise, without main diagonal
    upper_triangular_mask = torch.zeros_like(e_sample)
    indices = torch.triu_indices(row=e_sample.size(1), col=e_sample.size(2), offset=1)
    upper_triangular_mask[:, indices[0], indices[1]] = 1

    e_sample = e_sample * upper_triangular_mask
    e_sample = (e_sample + torch.transpose(e_sample, 1, 2))

    e_sample = torch.nn.functional.one_hot(e_sample.long(), num_classes=4)
    x_sample = torch.nn.functional.one_hot(x_sample.long(), num_classes=4)

    res = PlaceHolder(X=x_sample, E=e_sample, y=None).mask(mask)

    return res.X, res.E

    # smaple from uniform distribution

    # set lower triangular equal to upper
    # set lower triangular equal to upper

    print(e_sample.shape)
    e_sample = e_sample * torch.triu(torch.ones_like(e_sample), diagonal=1)
    # make onehot
    print(e_sample.shape)

    x_sample, e_sample = torch.nn.functional.one_hot(x_sample, num_classes=4), torch.nn.functional.one_hot(e_sample, num_classes=4)
    # set diagonal to zero
    print(e_sample.shape)


    # e_sample = e_sample * (~torch.eye(num_nodes).bool()).unsqueeze(0)





    print(e_sample[0, :, :, 0])
    quit()





    x, e = torch.randint(0, 4, size=(x_1.shape[0], x_1.shape[1])), torch.randint(0, 4, size=(
    e_1.shape[0], e_1.shape[1], e_1.shape[2]))

    theta_x_0, theta_e_0 = torch.nn.functional.one_hot(x_0, num_classes=4), torch.nn.functional.one_hot(edge_index_0,
                                                                                                        num_classes=4)
