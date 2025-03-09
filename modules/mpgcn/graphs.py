import numpy as np

class Graph():
    def __init__(self, form, labeling, part='body', max_hop=10, inter_link=False, base = 0, dilation=1, hop = 1, **kwargs):
        self.form = form

        self.part = part
        self.inter_link = inter_link
        self.base = base
        self.max_hop = max_hop
        self.dilation = dilation
        self.hop = hop
        
        if labeling not in ['spatial', 'zeros', 'ones', 'eye', 'intra-inter', 'distance']:
            raise ValueError('Error: Do NOT exist this graph labeling: {}!'.format(self.labeling))
        self.labeling = labeling

        # get edges
        self.num_node, self.num_person, self.edge, self.connect_joint, self.parts, self.center = self._get_edge()

        # get adjacency matrix
        self.A = self._get_adjacency()
        
    def __str__(self):
        return self.A

    def _get_edge(self):
        form = self.form.split('-')
        graph_base = form[0]
        num_person = 1 if len(form) == 1 else int(form[1])
        
        if graph_base == 'coco':
            num_node = 17
            neighbor_link = [(15, 13), (13, 11), (16, 14), (14, 12), (11, 5), (12, 6),
                                (9, 7), (7, 5), (10, 8), (8, 6), (5, 0), (6, 0),
                                (1, 0), (3, 1), (2, 0), (4, 2)]
            center = 0
            connect_joint = np.array(
                [0, 0, 0, 1, 2, 0, 0, 5, 6, 7, 8, 0, 0, 11, 12, 13, 14])
            parts = [
                np.array([5, 7, 9]),      # left_arm
                np.array([6, 8, 10]),     # right_arm
                np.array([11, 13, 15]),   # left_leg
                np.array([12, 14, 16]),    # right_leg
                np.array([0, 1, 2, 3, 4])  # head
            ]
        elif graph_base == 'coco_ball' or graph_base == 'coco_net':
            num_node = 17 + 1
            neighbor_link = [(15, 13), (13, 11), (16, 14), (14, 12), (11, 5), (12, 6),
                                (9, 7), (7, 5), (10, 8), (8, 6), (5, 0), (6, 0),
                                (1, 0), (3, 1), (2, 0), (4, 2), (9, 17), (10, 17)]
            center = 0
            connect_joint = np.array(
                [0, 0, 0, 1, 2, 0, 0, 5, 6, 7, 8, 0, 0, 11, 12, 13, 14, 10])
            if self.part == 'body':
                parts = [
                    np.array([5, 7, 9]),       # left_arm
                    np.array([6, 8, 10]),      # right_arm
                    np.array([11, 13, 15]),    # left_leg
                    np.array([12, 14, 16]),    # right_leg
                    np.array([0, 1, 2, 3, 4]), # head
                    np.array([17])             # object
                ]
            elif self.part == 'person':
                parts = [
                    np.arange(num_node-1), # body
                    np.array([num_node-1]) # object
                ]
        elif graph_base == 'coco_ball_net':
            num_node = 17 + 2
            neighbor_link = [(15, 13), (13, 11), (16, 14), (14, 12), (11, 5), (12, 6),
                                (9, 7), (7, 5), (10, 8), (8, 6), (5, 0), (6, 0),
                                (1, 0), (3, 1), (2, 0), (4, 2), (9, 17), (10, 17), (17, 18)]
            center = 0
            connect_joint = np.array(
                [0, 0, 0, 1, 2, 0, 0, 5, 6, 7, 8, 0, 0, 11, 12, 13, 14, 10, 17])
            if self.part == 'body':
                parts = [
                    np.array([5, 7, 9]),       # left_arm
                    np.array([6, 8, 10]),      # right_arm
                    np.array([11, 13, 15]),    # left_leg
                    np.array([12, 14, 16]),    # right_leg
                    np.array([0, 1, 2, 3, 4]), # head
                    np.array([17]),         # object
                    np.array([18]),
                ]
            elif self.part == 'person':
                parts = [
                    np.arange(num_node-2), # body
                    np.array([num_node-2]), # object
                    np.array([num_node-1])
                ]
        elif graph_base == 'openpose':
            num_node = 25
            neighbor_link = [(0, 1), (0, 15), (0, 16), (15, 17), (16, 18),
                                (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7),
                                (1, 8), (8, 9), (9, 10), (10,
                                                        11), (11, 24), (11, 22), (22, 23),
                                (8, 12), (12, 13), (13, 14), (14,
                                                            21), (14, 19), (19, 20)
                                ]
            connect_joint = np.array(
                [1, 1, 1, 2, 3, 1, 5, 6, 1, 8, 9, 10, 8, 12, 13, 0, 0, 15, 16, 14, 19, 14, 11, 22, 11])
            parts = [
                np.array([5, 6, 7]),                 # left_arm
                np.array([2, 3, 4]),                 # right_arm
                np.array([9, 10, 11, 22, 23, 24]),   # left_leg
                np.array([12, 13, 14, 19, 20, 21]),  # right_leg
                np.array([0, 1, 8, 15, 16, 17, 18])  # torso
            ]
            center = 1

        # producing multi-person graph 
        if num_person > 1:
            center = []
            neighbor_link_intra_nperson = []
            neighbor_link_inter_nperson = []
            connect_joint_nperson = []
            parts_nperson = []
            for i in range(num_person):
                center.extend([i*num_node]*num_node)
                for x in connect_joint:
                    connect_joint_nperson.append(x+i*num_node)
                for x, y in neighbor_link:
                    neighbor_link_intra_nperson.append(
                        (x+i*num_node, y+i*num_node))
                if self.part == 'body':
                    for p in range(len(parts)):
                        parts_nperson.append(parts[p]+i*num_node)
                elif self.part == 'person':
                    parts_nperson.append(np.arange(i*num_node, (i+1)*num_node))
            
            # add inter-person edges
            if self.inter_link == 'linear':
                for i in range(1, num_person):
                    neighbor_link_inter_nperson.append((i*num_node+self.base, (i-1)*num_node+self.base))
            elif self.inter_link == 'full':
                for i in range(num_person):
                    for j in range(i+1, num_person):
                        neighbor_link_inter_nperson.append((i*num_node+self.base, j*num_node+self.base))
            elif self.inter_link == 'star':
                for i in range(1, num_person):
                    neighbor_link_inter_nperson.append((self.base, i*num_node+self.base))
            elif self.inter_link == 'pairwise':
                h, d, m = self.hop, self.dilation, num_person
                for base in self.base if type(self.base) == list else [self.base]:
                    for i in range(h, m, d):
                        neighbor_link_inter_nperson.append((i*num_node+base, (i-self.hop)*num_node+base))
                    
            num_node *= num_person
            neighbor_link = neighbor_link_intra_nperson + neighbor_link_inter_nperson
            connect_joint = connect_joint_nperson
            parts = parts_nperson
        
        self_link = [(i, i) for i in range(num_node)]
        edge = self_link + neighbor_link

        if type(center) == int:
            center = [center] * num_node
        return num_node, num_person, edge, connect_joint, parts, center

    def _get_hop_distance(self):
        A = np.zeros((self.num_node, self.num_node))
        for i, j in self.edge:
            A[j, i] = 1
            A[i, j] = 1
        self.oA = A
        hop_dis = np.zeros((self.num_node, self.num_node)) + np.inf
        transfer_mat = [np.linalg.matrix_power(
            A, d) for d in range(self.max_hop + 1)]
        arrive_mat = (np.stack(transfer_mat) > 0)
        for d in range(self.max_hop, -1, -1):
            hop_dis[arrive_mat[d]] = d
        return hop_dis

    def _get_adjacency(self):

        if self.labeling == 'distance':
            hop_dis = self._get_hop_distance()
            valid_hop = range(0, self.max_hop + 1)
            adjacency = np.zeros((self.num_node, self.num_node))
            for hop in valid_hop:
                adjacency[hop_dis == hop] = 1
            normalize_adjacency = self._normalize_digraph(adjacency)

            A = np.zeros((len(valid_hop), self.num_node, self.num_node))
            for i, hop in enumerate(valid_hop):
                A[i][hop_dis == hop] = normalize_adjacency[hop_dis == hop]

        elif self.labeling == 'spatial':
            hop_dis = self._get_hop_distance()
            valid_hop = range(0, self.max_hop + 1)
            adjacency = np.zeros((self.num_node, self.num_node))
            for hop in valid_hop:
                adjacency[hop_dis == hop] = 1
            normalize_adjacency = self._normalize_digraph(adjacency)

            A = []
            for hop in valid_hop:
                a_root = np.zeros((self.num_node, self.num_node))
                a_close = np.zeros((self.num_node, self.num_node))
                a_further = np.zeros((self.num_node, self.num_node))
                for i in range(self.num_node):
                    for j in range(self.num_node):
                        if hop_dis[j, i] == hop:
                            # if hop_dis[j, self.center] == np.inf or hop_dis[i, self.center] == np.inf:
                            #     continue
                            if hop_dis[j, self.center[j]] == hop_dis[i, self.center[i]]:
                                a_root[j, i] = normalize_adjacency[j, i]
                            elif hop_dis[j, self.center[j]] > hop_dis[i, self.center[i]]:
                                a_close[j, i] = normalize_adjacency[j, i]
                            else:
                                a_further[j, i] = normalize_adjacency[j, i]
                if hop == 0:
                    A.append(a_root)
                else:
                    A.append(a_root + a_close)
                    A.append(a_further)
            A = np.stack(A)

        elif self.labeling == 'intra-inter':
            form = self.form.split('-')
            num_person = 1 if len(form) == 1 else int(form[1])
            if num_person == 1:
                raise ValueError('Error: intra-inter labeling requires m > 1!')
              
            hop_dis = self._get_hop_distance()
            valid_hop = range(0, self.max_hop + 1)
            adjacency = np.zeros((self.num_node, self.num_node))
            for hop in valid_hop:
                adjacency[hop_dis == hop] = 1
            normalize_adjacency = self._normalize_digraph(adjacency)

            def find_hop_edges(hop):
                indices = []  
                for i in range(self.num_node):  
                    for j in range(self.num_node):  
                        if hop_dis[i][j] == hop:  
                            indices.append((i, j))
                return indices
            A = []
            for hop in valid_hop:
                a_intra = np.zeros((self.num_node, self.num_node))
                a_inter = np.zeros((self.num_node, self.num_node))
                for src, dst in find_hop_edges(hop):
                    v = self.num_node // num_person
                    if src // v == dst // v:
                        a_intra[src, dst] = normalize_adjacency[src, dst]
                    else:
                        a_inter[src, dst] = normalize_adjacency[src, dst]
                if hop == 0:
                    A.append(a_intra)
                else:
                    A.append(a_intra)
                    A.append(a_inter)
            A = np.stack(A)
            
        elif self.labeling == 'zeros':
            valid_hop = range(0, self.max_hop + 1)
            A = np.zeros((len(valid_hop), self.num_node, self.num_node))

        elif self.labeling == 'ones':
            valid_hop = range(0, self.max_hop + 1)
            A = np.ones((len(valid_hop), self.num_node, self.num_node))
            for i in range(len(valid_hop)):
                A[i] = self._normalize_digraph(A[i])

        elif self.labeling == 'eye':
            valid_hop = range(0, self.max_hop + 1)
            A = np.zeros((len(valid_hop), self.num_node, self.num_node))
            for i in range(len(valid_hop)):
                A[i] = self._normalize_digraph(
                    np.eye(self.num_node, self.num_node))

        return A

    def _normalize_digraph(self, A):
        Dl = np.sum(A, 0)
        num_node = A.shape[0]
        Dn = np.zeros((num_node, num_node))
        for i in range(num_node):
            if Dl[i] > 0:
                Dn[i, i] = Dl[i]**(-1)
        AD = np.dot(A, Dn)
        return AD
