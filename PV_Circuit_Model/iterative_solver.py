'''
due to the coarseness of the element I-Vs, interpolation doesn't always
yield high accuracy in the operating point of a CircuitGroup
To restore high accuracy, we further use an iterative solver that solves the system of equations
related to the voltages of the nodes
'''

from scipy.sparse import coo_matrix
from scipy.sparse.linalg import factorized
from scipy.sparse.linalg import spsolve, cg, splu, lsqr
import numpy as np

def assign_nodes(circuit_group,node_count=0,scale=1.0):
    circuit_group.aux["scale"] = scale
    if hasattr(circuit_group,"is_cell"):
        circuit_group.aux["scale"] = circuit_group.area
    if node_count==0:
        circuit_group.aux["neg_node"] = node_count
        circuit_group.aux["pos_node"] = node_count + 1
        node_count += 2
    if hasattr(circuit_group, "is_circuit_group"):
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
    if hasattr(circuit_group, "is_multi_junction_cell"):
        for i, cell in enumerate(circuit_group.cells):
            PC_diodes = cell.findElementType("PhotonCouplingDiode")
            if len(PC_diodes)>0 and i>0:
                for PC_diode in PC_diodes:
                    PC_diode.aux["current_pos_node"] = circuit_group.cells[i-1].diode_branch.aux["neg_node"]
                    PC_diode.aux["current_neg_node"] = circuit_group.cells[i-1].diode_branch.aux["pos_node"]
    return node_count

def iterative_solve(circuit_group,V=None,I=None,starting_guess=None):
    # assumes assign_nodes has been done
    if starting_guess is not None:
        X = starting_guess.copy()
    else:
        # starting guess is just the operating point of the nodes
        net_list = []
        Y = []
        count = 0
        max_node_num = 0
        elements = circuit_group.findElementType("CircuitElement")
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
        MTM = M.T @ M
        MTY = M.T @ Y
        X = spsolve(MTM, MTY)
        
    X[circuit_group.aux["neg_node"]] = 0.0
    if V is not None:
        X[circuit_group.aux["pos_node"]] = V
        circuit_group.operating_point[0] = V 
    else:
        X[circuit_group.aux["pos_node"]] = circuit_group.operating_point[0]
        circuit_group.operating_point[1] = I

    best_X = None
    best_RMS = None
    best_net_I = None
    for iteration in range(10):
        elements = circuit_group.findElementType("CircuitElement")
        net_I_list = []
        dI_dV_list = []
        max_node_num = 0
        for element in elements:
            # current going out of node is positive
            V_ = X[element.aux["pos_node"]]-X[element.aux["neg_node"]]
            I_ = element.calc_I(V_)*element.aux["scale"]
            dI_dV = element.calc_dI_dV(V_)*element.aux["scale"]
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
        if best_RMS is None or RMS < best_RMS:
            best_RMS = RMS
            best_X = X
            best_net_I = net_I
        if RMS < 1e-10 or iteration==9:
            break
        M = M[indices][:, indices]
        delta_X = np.zeros((max_node_num+1,))
        rhs = -np.ravel(I_error)   

        delta_X[indices] = spsolve(M, rhs) # direct solve
        # delta_X[indices], _ = cg(M, rhs, atol=1e-12, maxiter=500) # CG
        # delta_X[indices] = lsqr(M, rhs, damp=0.0, atol=1e-12, btol=1e-12, iter_lim=500)[0] # LSQR
        new_X = X + delta_X
        
        max_jump = 0
        forward_diodes = circuit_group.findElementType("ForwardDiode")
        for element in forward_diodes:
            old_V_ = X[element.aux["pos_node"]]-X[element.aux["neg_node"]]
            new_V_ = new_X[element.aux["pos_node"]]-new_X[element.aux["neg_node"]]
            max_jump = max(max_jump, new_V_-old_V_)
        
        reverse_diodes = circuit_group.findElementType("ReverseDiode")
        for element in reverse_diodes:
            old_V_ = X[element.aux["pos_node"]]-X[element.aux["neg_node"]]
            new_V_ = new_X[element.aux["pos_node"]]-new_X[element.aux["neg_node"]]
            max_jump = max(max_jump, old_V_-new_V_)

        if max_jump > 0.3:
            delta_X *= 0.3/max_jump
        X += delta_X
    if V is None:
        circuit_group.operating_point[0] = best_X[circuit_group.aux["pos_node"]] - best_X[circuit_group.aux["neg_node"]]
    if I is None:
        circuit_group.operating_point[1] = 0.5*(best_net_I[int(circuit_group.aux["pos_node"])]-best_net_I[int(circuit_group.aux["neg_node"])])
    return circuit_group.operating_point[:2]
    # return M, X, RMS, 0.5*(net_I[int(circuit_group.aux["pos_node"])]-net_I[int(circuit_group.aux["neg_node"])])

# def get_dI_dV(circuit_group,M):
#     rows = M.shape[0]
#     include_ = np.ones((rows,),dtype=bool)
#     include_[int(circuit_group.aux["neg_node"])] = False
#     include_[int(circuit_group.aux["pos_node"])] = False
#     indices = np.where(include_==True)[0]
#     bias_V = np.zeros((rows,))
#     bias_V[int(circuit_group.aux["pos_node"])] = 1.0
#     Y = M @ bias_V
#     X = bias_V.copy()
#     X[indices] = spsolve(M[indices][:, indices], -np.ravel(Y[indices]))
#     net_I = M @ X
#     return 0.5*(net_I[int(circuit_group.aux["pos_node"])]-net_I[int(circuit_group.aux["neg_node"])])
