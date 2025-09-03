from PV_Circuit_Model.circuit_model import *
from PV_Circuit_Model.multi_junction_cell import *
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import factorized
from scipy.sparse.linalg import spsolve, cg, splu, lsqr

def assign_nodes(circuit_group,node_count=0,scale=1.0):
    circuit_group.aux["scale"] = scale
    if isinstance(circuit_group,Cell) and not isinstance(circuit_group,MultiJunctionCell):
        circuit_group.aux["scale"] = circuit_group.area
    if node_count==0:
        circuit_group.aux["neg_node"] = node_count
        circuit_group.aux["pos_node"] = node_count + 1
        node_count += 2
    if isinstance(circuit_group,CircuitGroup):
        for i, element in enumerate(circuit_group.subgroups):
            if i==0 or circuit_group.connection == "parallel":
                element.aux["neg_node"] = circuit_group.aux["neg_node"]
            else:
                element.aux["neg_node"] = node_count 
                node_count += 1
            if i==len(circuit_group.subgroups)-1 or circuit_group.connection == "parallel":
                element.aux["pos_node"] = circuit_group.aux["pos_node"]
            else:
                element.aux["pos_node"] = node_count
                # do not increment node count because next element neg node uses it
        for element in circuit_group.subgroups:
            node_count = assign_nodes(element,node_count=node_count,scale=circuit_group.aux["scale"])
    # PC diodes
    if isinstance(circuit_group,MultiJunctionCell):
        for i, cell in enumerate(circuit_group.cells):
            PC_diodes = cell.findElementType(PhotonCouplingDiode)
            if len(PC_diodes)>0 and i>0:
                for PC_diode in PC_diodes:
                    PC_diode.aux["current_pos_node"] = circuit_group.cells[i-1].diode_branch.aux["neg_node"]
                    PC_diode.aux["current_neg_node"] = circuit_group.cells[i-1].diode_branch.aux["pos_node"]
    return node_count

def construct_matrix(circuit_group,V=None,I=None,starting_guess=None):
    if starting_guess is not None:
        X = starting_guess.copy()
    else:
        net_list = []
        Y = []
        count = 0
        max_node_num = 0
        elements = circuit_group.findElementType(CircuitElement)
        for element in elements:
            net_list.append([count,element.aux["pos_node"],1])
            net_list.append([count,element.aux["neg_node"],-1])
            Y.append(element.operating_point[0])
            count += 1
            max_node_num = max(max_node_num, element.aux["pos_node"], element.aux["neg_node"])
        net_list.append([count,circuit_group.aux["neg_node"],1])
        count += 1
        Y.append(0.0)
        net_list = np.array(net_list)
        M = coo_matrix((net_list[:,2], (net_list[:,0].astype(int), net_list[:,1].astype(int))), shape=(count, max_node_num+1))
        M = M.tocsc().astype('float64', copy=False)
        Y = np.array(Y).squeeze()
        X = lsqr(M, Y, damp=0.0, atol=1e-12, btol=1e-12, iter_lim=2000)[0]
        
    X[circuit_group.aux["neg_node"]] = 0.0
    if V is not None:
        X[circuit_group.aux["pos_node"]] = V
    else:
        X[circuit_group.aux["pos_node"]] = circuit_group.operating_point[0]

    for iteration in range(10):
        elements = circuit_group.findElementType(CircuitElement)
        net_I_list = []
        dI_dV_list = []
        max_node_num = 0
        for element in elements:
            # current going out of node is positive
            V = X[element.aux["pos_node"]]-X[element.aux["neg_node"]]
            I_ = element.calc_I(V)*element.aux["scale"]
            dI_dV = element.calc_dI_dV(V)*element.aux["scale"]
            net_I_list.append([element.aux["neg_node"],-I_])
            net_I_list.append([element.aux["pos_node"],I_])
            dI_dV_list.append([element.aux["neg_node"],element.aux["pos_node"],-dI_dV])
            dI_dV_list.append([element.aux["pos_node"],element.aux["neg_node"],-dI_dV])
            dI_dV_list.append([element.aux["neg_node"],element.aux["neg_node"],dI_dV])
            dI_dV_list.append([element.aux["pos_node"],element.aux["pos_node"],dI_dV])
            if "current_pos_node" in element.aux:
                net_I_list.append([element.aux["current_neg_node"],-I_])
                net_I_list.append([element.aux["current_pos_node"],I_])
                dI_dV_list.append([element.aux["current_neg_node"],element.aux["pos_node"],-dI_dV])
                dI_dV_list.append([element.aux["current_pos_node"],element.aux["neg_node"],-dI_dV])
                dI_dV_list.append([element.aux["current_neg_node"],element.aux["neg_node"],dI_dV])
                dI_dV_list.append([element.aux["current_pos_node"],element.aux["pos_node"],dI_dV])
            max_node_num = max(max_node_num, element.aux["pos_node"], element.aux["neg_node"])
        net_I_list = np.array(net_I_list)
        dI_dV_list = np.array(dI_dV_list)
        M = coo_matrix((dI_dV_list[:,2], (dI_dV_list[:,0].astype(int), dI_dV_list[:,1].astype(int))), shape=(max_node_num+1, max_node_num+1))
        net_I = coo_matrix((net_I_list[:,1], (net_I_list[:,0].astype(int), np.zeros_like(net_I_list[:,0],dtype=int))), shape=(max_node_num+1, 1))
        M = M.tocsc().astype('float64', copy=False)
        net_I.sum_duplicates()
        net_I = net_I.toarray().squeeze()
        I_error = net_I.copy()
        if I is not None:
            I_error[int(circuit_group.aux["pos_node"])] -= circuit_group.operating_point[1]
        include_ = np.ones((max_node_num+1,),dtype=bool)
        include_[int(circuit_group.aux["neg_node"])] = False
        if I is None:
            include_[int(circuit_group.aux["pos_node"])] = False
        indices = np.where(include_==True)[0]
        I_error = I_error[indices]
        RMS = np.sqrt(np.mean(I_error**2))
        if RMS < 1e-12 or iteration==9:
            break
        M = M[indices][:, indices]
        # solve_M = factorized(M)     # LU factorization cached
        delta_X = np.zeros((max_node_num+1,))
        # delta_X[indices] = solve_M(-np.ravel(I_error))
        delta_X[indices] = spsolve(M, -np.ravel(I_error)) 
        X += delta_X
    return M, X, RMS, 0.5*(net_I[int(circuit_group.aux["pos_node"])]-net_I[int(circuit_group.aux["neg_node"])])

def get_dI_dV(circuit_group,M):
    rows = M.shape[0]
    include_ = np.ones((rows,),dtype=bool)
    include_[int(circuit_group.aux["neg_node"])] = False
    include_[int(circuit_group.aux["pos_node"])] = False
    indices = np.where(include_==True)[0]
    bias_V = np.zeros((rows,))
    bias_V[int(circuit_group.aux["pos_node"])] = 1.0
    Y = M @ bias_V
    X = bias_V.copy()
    X[indices] = spsolve(M[indices][:, indices], -np.ravel(Y[indices]))
    net_I = M @ X
    return 0.5*(net_I[int(circuit_group.aux["pos_node"])]-net_I[int(circuit_group.aux["neg_node"])])
