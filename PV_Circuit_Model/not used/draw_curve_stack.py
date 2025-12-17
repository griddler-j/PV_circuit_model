
from PV_Circuit_Model.circuit_model import *
from PV_Circuit_Model.device import Cell

# This is only an approx way to visualize the curve stacking, by no means rigorous!  
# Idea is to just draw the 4th quadrant of the curve stretched within the bounds 
def draw_curve_stack(self,max_depth=3,left_bound_curve=None,right_bound_curve=None,origin_V=0,origin_I=0,is_root=True,group_origin_V=None):
    fills = []
    plots = []
    if isinstance(self,CircuitElement): # don't draw elements
        return [], []
    if max_depth==0 or isinstance(self,Cell): # draw at cell level
        Voc = self.get_Voc()
        V_sample = np.linspace(0,Voc*1.5,5000) 
        find_ = np.where((V_sample>=self.IV_V[0]) & (V_sample<=self.IV_V[-1]))[0]
        V_sample = V_sample[find_]
        I_sample = interp_(V_sample,self.IV_V,self.IV_I)
        V_final = V_sample + origin_V
        I_final = I_sample + origin_I
        if left_bound_curve is not None:
            V_final += interp_(I_final, left_bound_curve[1,:], left_bound_curve[0,:]) - origin_V
        if right_bound_curve is not None:
            V_final = np.minimum(V_final,interp_(I_final, right_bound_curve[1,:], right_bound_curve[0,:]))
        if group_origin_V is not None:
            find2_ = np.where(V_final>group_origin_V)[0]
            return [[V_final,I_final,Voc]], [[V_final[find2_],I_final[find2_]]]
        return [[V_final,I_final,Voc]], []
    if self.connection=="series":
        Iscs = []
        I_domain = None
        for item_ in self.subgroups:
            if isinstance(item_,CircuitElement):
                Iscs.append(0)
            else:
                if I_domain is None:
                    I_domain = np.linspace(np.min(item_.IV_I),np.max(item_.IV_I),20000) 
                Iscs.append(item_.get_Isc())
        Iscs = np.array(Iscs)
        indices = np.argsort(Iscs)
        if left_bound_curve is None:
            left_bound_curve_ = np.array([np.zeros_like(I_domain),I_domain])
        else:
            left_bound_curve_ = left_bound_curve.copy()
        Voc_sum = 0
        for index in indices:
            item = self.subgroups[index]
            added_V = interp_(left_bound_curve_[1,:],item.IV_I+origin_I,item.IV_V)
            right_bound_curve_ = left_bound_curve_.copy()
            right_bound_curve_[0,:] += added_V
            if isinstance(item,CircuitGroup):
                new_origin_V = interp_(origin_I,right_bound_curve_[1,:],right_bound_curve_[0,:])
                min_right_bound_ = right_bound_curve_.copy()
                if right_bound_curve is not None:
                    min_right_bound_[0,:] = np.minimum(min_right_bound_[0,:],interp_(min_right_bound_[1,:],right_bound_curve[1,:],right_bound_curve[0,:]))

                is_last = index==indices[-1]
                group_origin_V = None
                if is_last:
                    group_origin_V = origin_V
                fills_, plots_ = item.draw_curve_stack(max_depth=max_depth-1, left_bound_curve=left_bound_curve_, right_bound_curve=min_right_bound_,
                                            origin_V=new_origin_V,origin_I=origin_I,is_root=False,group_origin_V=group_origin_V)
                fills.extend(fills_)
                plots.extend(plots_)
                Voc_sum += item.get_Voc()
            left_bound_curve_[0,:] += added_V
    else: # parallel
        Vocs = []
        V_domain = None
        for item_ in self.subgroups:
            if isinstance(item_, CircuitElement):
                Vocs.append(0)
            else:
                if V_domain is None:
                    V_domain = np.linspace(np.min(item_.IV_V),np.max(item_.IV_V),20000) 
                Vocs.append(item_.get_Voc())
        Vocs = np.array(Vocs)
        indices = np.argsort(Vocs)
        Isc_sum = 0
            
        for index in indices:
            item = self.subgroups[index]
            if isinstance(item, CircuitGroup):
                new_origin_I = origin_I-Isc_sum
                if left_bound_curve is not None:
                    new_origin_V = interp_(new_origin_I,left_bound_curve[1,:],left_bound_curve[0,:])
                else:
                    new_origin_V = 0
                is_last = index==indices[-1]
                group_origin_V = None
                if is_last:
                    group_origin_V = new_origin_V
                fills_, plots_ = item.draw_curve_stack(max_depth=max_depth-1, left_bound_curve=left_bound_curve, right_bound_curve=right_bound_curve,
                                            origin_V=new_origin_V,origin_I=new_origin_I,is_root=False,group_origin_V=group_origin_V)
                fills.extend(fills_)
                plots.extend(plots_)
                Isc_sum += item.get_Isc() # mindful that it is flipped 
    if is_root:
        min_Voc = 10000
        max_Voc = 0
        for fill_ in reversed(fills):
            Voc = fill_[2]
            min_Voc = min(min_Voc,Voc)
            max_Voc = max(max_Voc,Voc)
        for fill_ in reversed(fills):
            V_final = fill_[0]
            I_final = fill_[1]
            Voc = fill_[2]
            Voc_norm = np.clip((Voc - min_Voc) / (max_Voc - min_Voc + 1e-12), 0, 1)

            # Pick a nice scientific colormap
            cmap = cm.plasma   # also great: plasma, inferno, turbo, magma
            color = cmap(Voc_norm)  # RGBA tuple

            plt.fill_between(V_final, -I_final, 0, color=color)
        for plot_ in reversed(plots):
            V_final = plot_[0]
            I_final = plot_[1]
            plt.plot(V_final, -I_final, color="black",linewidth=5)
        Voc = self.get_Voc()
        Isc = self.get_Isc()
        plt.plot(self.IV_V,-self.IV_I,color="black",linewidth=5)
        plt.xlim((0,Voc*1.1))
        plt.ylim((0,Isc*1.1))
        plt.show()
    return fills, plots

CircuitGroup.draw_curve_stack = draw_curve_stack     
            