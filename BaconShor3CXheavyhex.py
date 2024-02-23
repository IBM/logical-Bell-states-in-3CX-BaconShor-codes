# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

from qiskit import QuantumCircuit
from qiskit_aer.noise import depolarizing_error, pauli_error
from typing import Union

import numpy as np
import matplotlib.pyplot as plt

class BS3CXheavyhex:
    def __init__(self, d:int = 2, T:int = 1, logical_prep:str = ['Z','Z'],logical_meas:str = ['Z','Z'],
                log_gate: Union[str,list] = '', log_round: Union[int,list] = -1,
                num_qubits: int = 127, offset: complex = -2+6j, 
                anc_reset: bool = True, virt_link: bool = False,
                CX_schedule: list = ['3CX0', '3CX1', '3CX2', '3CX3', 'BSX0', 'BSX1', 'BSZ0', 'BSZ1'],
                CXerror:float = 0, Rerror: Union[float,list] = [0,0], singleQerror:float = 0, idleerror:float = 0):
        
        if isinstance(log_round,int):
            log_round = [log_round]
        if isinstance(log_gate,str):
            log_gate = [log_gate]*len(log_round)
        if not isinstance(Rerror,list):
            Rerror = [Rerror]*2

        self.CXerror = CXerror
        self.Rerror = Rerror
        self.singleQerror = singleQerror
        self.d = d
        self.T = T
        self.logical_prep = logical_prep
        self.logical_meas = logical_meas
        if isinstance(log_round,int):
            self.log_round = [log_round]
        else:
            self.log_round = log_round
        if isinstance(log_round,str):
            self.log_gate = [log_gate]
        else:
            self.log_gate = log_gate
        self.virt_link = virt_link

        qubit_coordinates3CX = [offset+k*(1+1j) + i*(1-1j) for k in range(2, 2*d+1, 2) for i in range(2, 2*d+1, 2)]
        ancilla_coordinates3CX = [offset+k*(1+1j) + i*(1-1j) + 2j for k in range(2, 2*d+1, 2) for i in range(2, 2*d+1, 2) if k!=2*d or i !=2]
        qubit_coordinatesBS = [q+1j for q in qubit_coordinates3CX]
        ancilla_coordinatesBS = [offset+1j+k*(1+1j) + i*(1-1j) for k in range(2, 2*d+1) 
                                 for i in range(2, 2*d+1) 
                                 if (k%2==0 or i%2==0)
                                 and (offset+1j+k*(1+1j) + i*(1-1j) not in qubit_coordinatesBS)]

        # preparing the dictionary
        self.q2i3CX = {q: i + d*(i//d) for i, q in enumerate(self.sorted_complex(qubit_coordinates3CX))}
        self.a2i3CX = {a: i - d + d*(i//d) for i, a in enumerate(self.sorted_complex(ancilla_coordinates3CX),start=d)}
        self.q2iBS = {q: i for i, q in enumerate(self.sorted_complex(qubit_coordinatesBS),start=2*d**2-1)}
        self.a2iBS = {a: i for i, a in enumerate(self.sorted_complex(ancilla_coordinatesBS),start=3*d**2-1)}
        
        i2q3CX = {v: k for k,v in  self.q2i3CX.items()}
        

        def rot_qpos(qpos,dir=1):
            if dir==1:
                return (qpos-i2q3CX[0])*(1-1j)+2+2j
            elif dir==-1:
                return (qpos-2-2j)*(1+1j)/2+i2q3CX[0]

        qubit_pos3CX = [rot_qpos(k) for k in  self.q2i3CX.keys()]
        ancilla_pos3CX = [rot_qpos(k) for k in  self.a2i3CX.keys()]
        self.qubit_pos3CX = qubit_pos3CX

        qubit_posBS = [rot_qpos(k) for k in  self.q2iBS.keys()]
        ancilla_posBS = [rot_qpos(k) for k in  self.a2iBS.keys()]
        self.qubit_posBS = qubit_posBS

        if num_qubits == -1:
            heavyHEX_dict = self.q2i3CX.copy()
            heavyHEX_dict.update(self.a2i3CX)
            heavyHEX_dict.update(self.q2iBS)
            heavyHEX_dict.update(self.a2iBS)
            if 'LS' in log_gate[-1]:
                heavyHEX_dict.update({rot_qpos(2+k*1j,-1)-1: i for i,k in enumerate(range(2,4*d+2,4), start = len(heavyHEX_dict))})
                heavyHEX_dict.update({rot_qpos(k+2j,-1)+1: i for i,k in enumerate(range(2,4*d+2,4), start = len(heavyHEX_dict))})
            self.num_qubits = len(heavyHEX_dict)
        else:
            heavyHEX_dict = self.buildHeavyHEX(num_qubits)
            self.num_qubits = num_qubits
        self.heavyHEX_dict = heavyHEX_dict
        self.vertex_qubits = [self.heavyHEX_dict[pos] for pos in list(self.q2i3CX.keys())+list(self.a2i3CX.keys())] # for readout errors

        qubit_index_list3CX = [heavyHEX_dict[rot_qpos(qpos,-1)] for qpos in qubit_pos3CX]
        qubit_index_listBS = [heavyHEX_dict[rot_qpos(qpos,-1)] for qpos in qubit_posBS]

        self.qubit_index_list3CX = qubit_index_list3CX
        self.qubit_index_listBS = qubit_index_listBS

        if d%2 == 1:
            anc_posa = [a for a in ancilla_pos3CX if (a.real==4*d and a.imag%8==4) or (a.imag==4*d and a.real%8==0)]
            anc_posb = [a for a in ancilla_pos3CX if (a.real==4*d and a.imag%8==0) or (a.imag==4*d and a.real%8==4)]
        else:
            anc_posb = [a for a in ancilla_pos3CX if (a.real==4*d and a.imag%8==4) or (a.imag==4*d and a.real%8==0)]
            anc_posa = [a for a in ancilla_pos3CX if (a.real==4*d and a.imag%8==0) or (a.imag==4*d and a.real%8==4)]

        x_ancilla_pos3CX = [a for a in ancilla_pos3CX if (a.real-2+a.imag-2)%8 == 0]
        z_ancilla_pos3CX = [a for a in ancilla_pos3CX if (a.real-2+a.imag-2)%8 == 4]

        self.measure_x_qubit_pos_cycle0 = [anc_pos for anc_pos in ancilla_pos3CX 
                                    if (anc_pos.real<4*d and anc_pos in z_ancilla_pos3CX)
                                        or anc_pos.imag==4*d]
        measure_x_qubits_cycle0 = [heavyHEX_dict[rot_qpos(anc_pos,-1)] for anc_pos in self.measure_x_qubit_pos_cycle0]
        self.measure_z_qubit_pos_cycle0 = [anc_pos for anc_pos in ancilla_pos3CX 
                                    if (anc_pos.imag<4*d and anc_pos in x_ancilla_pos3CX)
                                        or anc_pos.real==4*d]
        measure_z_qubits_cycle0 = [heavyHEX_dict[rot_qpos(anc_pos,-1)] for anc_pos in self.measure_z_qubit_pos_cycle0]
        self.measure_x_qubit_pos_cycle1 = [anc_pos for anc_pos in ancilla_pos3CX 
                                    if (anc_pos.real<4*d and anc_pos in x_ancilla_pos3CX)
                                        or anc_pos.imag==4*d]
        measure_x_qubits_cycle1 = [heavyHEX_dict[rot_qpos(anc_pos,-1)] for anc_pos in self.measure_x_qubit_pos_cycle1]
        self.measure_z_qubit_pos_cycle1 = [anc_pos for anc_pos in ancilla_pos3CX 
                                    if (anc_pos.imag<4*d and anc_pos in z_ancilla_pos3CX)
                                        or anc_pos.real==4*d]
        measure_z_qubits_cycle1 = [heavyHEX_dict[rot_qpos(anc_pos,-1)] for anc_pos in self.measure_z_qubit_pos_cycle1]


        measure_x_qubit_posBS = [anc for anc in ancilla_posBS if anc.real%4==1]
        self.measure_x_qubit_posBS = measure_x_qubit_posBS
        measure_x_qubitsBS = [heavyHEX_dict[rot_qpos(anc_pos,-1)] for anc_pos in measure_x_qubit_posBS]
        measure_z_qubit_posBS = [anc for anc in ancilla_posBS if anc.imag%4==1]
        self.measure_z_qubit_posBS = measure_z_qubit_posBS
        measure_z_qubitsBS = [heavyHEX_dict[rot_qpos(anc_pos,-1)] for anc_pos in measure_z_qubit_posBS]


        pair_target_pos3CX_round0 = [[rot_qpos(anc_pos,-1), rot_qpos(anc_pos-2-2j,-1)] 
                                     for anc_pos in x_ancilla_pos3CX if anc_pos-2-2j in qubit_pos3CX and anc_pos not in anc_posb]
        pair_target_pos3CX_round0.extend([rot_qpos(anc_pos-2-2j,-1), rot_qpos(anc_pos,-1)] 
                                         for anc_pos in z_ancilla_pos3CX if anc_pos-2-2j in qubit_pos3CX and anc_pos not in anc_posb)
        pair_target_posBS_round0 = [[rot_qpos(anc_pos,-1),rot_qpos(anc_pos,-1)-1-1j] for anc_pos in measure_x_qubit_posBS]


        pair_target_pos3CX_round1 = [[rot_qpos(anc_pos,-1), rot_qpos(anc_pos+2-2j,-1)] 
                                     for anc_pos in x_ancilla_pos3CX if anc_pos+2-2j in qubit_pos3CX and anc_pos not in anc_posb]
        pair_target_pos3CX_round1.extend([rot_qpos(anc_pos-2+2j,-1), rot_qpos(anc_pos,-1)] 
                                         for anc_pos in z_ancilla_pos3CX if anc_pos-2+2j in qubit_pos3CX and anc_pos not in anc_posb)
        pair_target_posBS_round1 = [[rot_qpos(anc_pos,-1),rot_qpos(anc_pos,-1)+1+1j] for anc_pos in measure_x_qubit_posBS]


        pair_target_pos3CX_round2 = [[rot_qpos(anc_pos-2+2j,-1), rot_qpos(anc_pos,-1)] 
                                     for anc_pos in x_ancilla_pos3CX if anc_pos-2+2j in qubit_pos3CX and anc_pos not in  anc_posa]
        pair_target_pos3CX_round2.extend([rot_qpos(anc_pos,-1), rot_qpos(anc_pos+2-2j,-1)] 
                                         for anc_pos in z_ancilla_pos3CX if anc_pos+2-2j in qubit_pos3CX and anc_pos not in  anc_posa)
        pair_target_posBS_round2 = [[rot_qpos(anc_pos,-1)-1+1j,rot_qpos(anc_pos,-1)] for anc_pos in measure_z_qubit_posBS]


        pair_target_pos3CX_round3 = [[rot_qpos(anc_pos-2-2j,-1), rot_qpos(anc_pos,-1)] 
                                     for anc_pos in x_ancilla_pos3CX if anc_pos-2-2j in qubit_pos3CX and anc_pos not in  anc_posa]
        pair_target_pos3CX_round3.extend([rot_qpos(anc_pos,-1), rot_qpos(anc_pos-2-2j,-1)] 
                                         for anc_pos in z_ancilla_pos3CX if anc_pos-2-2j in qubit_pos3CX and anc_pos not in  anc_posa)
        pair_target_posBS_round3 = [[rot_qpos(anc_pos,-1)+1-1j,rot_qpos(anc_pos,-1)] for anc_pos in measure_z_qubit_posBS]


        self.pair_target_pos_dict = {'3CX0': np.transpose(pair_target_pos3CX_round0),
                                '3CX1': np.transpose(pair_target_pos3CX_round1),
                                '3CX2': np.transpose(pair_target_pos3CX_round2),
                                '3CX3': np.transpose(pair_target_pos3CX_round3),
                                'BSX0': np.transpose(pair_target_posBS_round0),
                                'BSX1': np.transpose(pair_target_posBS_round1),
                                'BSZ0': np.transpose(pair_target_posBS_round2),
                                'BSZ1': np.transpose(pair_target_posBS_round3)}

        # edge qubits
        edgeZ_qubit_pos3CX = [k+2j for k in range(2,4*d+2,4)]
        edgeX_qubit_pos3CX = [2+k*1j for k in range(2,4*d+2,4)]
        edgeZ_qubit_posBS = [k+2j+1+1j for k in range(2,4*d+2,4)]
        edgeX_qubit_posBS = [2+k*1j+1+1j for k in range(2,4*d+2,4)]
        if logical_meas[0] == 'Z':
            self.edge_qubit_pos3CX = edgeZ_qubit_pos3CX
        elif logical_meas[0] == 'X':
            self.edge_qubit_pos3CX = edgeX_qubit_pos3CX
        if logical_meas[1] == 'Z':
            self.edge_qubit_posBS = edgeZ_qubit_posBS
        elif logical_meas[1] == 'X':
            self.edge_qubit_posBS = edgeX_qubit_posBS

        self.edge_qubits3CX =  [heavyHEX_dict[rot_qpos(qpos,-1)] for qpos in self.edge_qubit_pos3CX]
        self.edge_qubitsBS =  [heavyHEX_dict[rot_qpos(qpos,-1)] for qpos in self.edge_qubit_posBS]

        #build circuit
        qc = QuantumCircuit(self.num_qubits,(2*d*(d-1)+d**2-1)*T+2*d**2+('LS' in log_gate[-1])*2*d)

        #initialize qubits and ancillas
        if self.num_qubits==-1:
            self.noisy_reset(qc,qubit_index_list3CX)
            self.noisy_reset(qc,qubit_index_listBS)
            self.noisy_reset(qc,measure_x_qubits_cycle0)
            self.noisy_reset(qc,measure_z_qubits_cycle0)
            self.noisy_reset(qc,measure_x_qubitsBS)
            self.noisy_reset(qc,measure_z_qubitsBS)
        if logical_prep[0] == 'X':
            self.noisy_h(qc,qubit_index_list3CX)
        if logical_prep[1] == 'X':
            self.noisy_h(qc,qubit_index_listBS)
        self.noisy_h(qc,measure_x_qubits_cycle0)
        self.noisy_h(qc,measure_x_qubitsBS)

        num_meas_per_round = d**2-1 + 2*d*(d-1)
        measuredict = {} 
        for time in range(T):
            if time in log_round:
                if log_gate[log_round.index(time)]=='CX':
                    for q1,q2 in zip(qubit_index_list3CX,qubit_index_listBS):
                        self.noisy_cx(qc,[q1,q2])
                elif log_gate[log_round.index(time)]=='XC':
                    for q1,q2 in zip(qubit_index_listBS,qubit_index_list3CX):
                        self.noisy_cx(qc,[q1,q2])
            # qc.barrier()

            if time%2==1:                
                for round_key in CX_schedule:
                    pair_target_pos = self.pair_target_pos_dict[round_key]
                    self.skip_CX(qc,pair_target_pos[0],None,pair_target_pos[1],None, version = 'b')
                if time!=T-1:
                    qc.barrier()

                self.noisy_h(qc,measure_x_qubits_cycle0)
                self.noisy_measure(qc,measure_x_qubits_cycle0, [time*num_meas_per_round+i for i in range(int(np.ceil((d**2-1)/2)))])
                self.noisy_measure(qc,measure_z_qubits_cycle0, [time*num_meas_per_round+int(np.ceil((d**2-1)/2))+i for i in range(int((d**2-1)/2))])
                if idleerror>0:
                    for q in qubit_index_list3CX:
                        qc.append(depolarizing_error(idleerror,1),[q])
                    for q in qubit_index_listBS:
                        qc.append(depolarizing_error(idleerror,1),[q])
                if time!= T-1:
                    if anc_reset:
                        self.noisy_reset(qc,measure_x_qubits_cycle0)
                        self.noisy_reset(qc,measure_z_qubits_cycle0)
                    self.noisy_h(qc,measure_x_qubits_cycle0)
                measuredict.update({
                    (self.measure_x_qubit_pos_cycle0[i],time): time*num_meas_per_round+i 
                    for i in range(int(np.ceil((d**2-1)/2)))
                    })
                measuredict.update({
                    (self.measure_z_qubit_pos_cycle0[i],time): time*num_meas_per_round+int(np.ceil((d**2-1)/2))+i 
                    for i in range(int((d**2-1)/2))
                    })
            
            elif time%2==0:
                for round_key in CX_schedule[::-1]:
                    pair_target_pos = self.pair_target_pos_dict[round_key]
                    self.skip_CX(qc,pair_target_pos[0],None,pair_target_pos[1],None, version = 'a')                
                if time!=T-1:
                    qc.barrier()
         
                self.noisy_h(qc,measure_x_qubits_cycle1)            
                self.noisy_measure(qc,measure_x_qubits_cycle1, [time*num_meas_per_round+i for i in range(int((d**2-1)/2))])
                self.noisy_measure(qc,measure_z_qubits_cycle1, [time*num_meas_per_round+int((d**2-1)/2)+i for i in range(int(np.ceil((d**2-1)/2)))])
                if idleerror>0:
                    for q in qubit_index_list3CX:
                        qc.append(depolarizing_error(idleerror,1),[q])
                    for q in qubit_index_listBS:
                        qc.append(depolarizing_error(idleerror,1),[q])
                if time!= T-1:
                    if anc_reset:
                        self.noisy_reset(qc,measure_x_qubits_cycle1)
                        self.noisy_reset(qc,measure_z_qubits_cycle1)
                    self.noisy_h(qc,measure_x_qubits_cycle1)
                measuredict.update({
                    (self.measure_x_qubit_pos_cycle1[i],time): time*num_meas_per_round+i 
                    for i in range(int((d**2-1)/2))
                    })
                measuredict.update({
                    (self.measure_z_qubit_pos_cycle1[i],time): time*num_meas_per_round+int((d**2-1)/2)+i 
                    for i in range(int(np.ceil((d**2-1)/2)))
                    })
            
            #BS measurements
        
            self.noisy_h(qc,measure_x_qubitsBS)            
            self.noisy_measure(qc,measure_x_qubitsBS, [time*num_meas_per_round+(d**2-1)+i for i in range(d*(d-1))])
            self.noisy_measure(qc,measure_z_qubitsBS, [time*num_meas_per_round+(d**2-1)+d*(d-1)+i for i in range(d*(d-1))])
            if time!= T-1:
                if anc_reset:
                    self.noisy_reset(qc,measure_x_qubitsBS)
                    self.noisy_reset(qc,measure_z_qubitsBS)
                self.noisy_h(qc,measure_x_qubitsBS)
            if time!=T-1:
                qc.barrier()
            measuredict.update({
                (measure_x_qubit_posBS[i],time): time*num_meas_per_round+(d**2-1)+i 
                for i in range(d*(d-1))
                })
            measuredict.update({
                (measure_z_qubit_posBS[i],time): time*num_meas_per_round+(d**2-1)+d*(d-1)+i 
                for i in range(d*(d-1))
                })

        self.LS_ancillas = []
        self.LS_anc_pos = []
        self.LSXX_pos = []
        self.LSYY_pos = []
        self.LSZZ_pos = []
        fake_anc_CX = -1
        if T in log_round:
            if log_gate[log_round.index(T)]=='CX':
                for q1,q2 in zip(qubit_index_list3CX,qubit_index_listBS):
                    self.noisy_cx(qc,[q1,q2])
            elif log_gate[log_round.index(T)]=='XC':
                for q1,q2 in zip(qubit_index_listBS,qubit_index_list3CX):
                    self.noisy_cx(qc,[q1,q2])
            elif log_gate[log_round.index(T)][:2]=='LS':
                if 'Z' in log_gate[log_round.index(T)]:
                    for qpos3CX,qposBS in zip(edgeZ_qubit_pos3CX,edgeZ_qubit_posBS):
                        qpos3CX_hhx,qposBS_hhx = rot_qpos(qpos3CX,-1),rot_qpos(qposBS,-1)
                        if qpos3CX_hhx+1 in heavyHEX_dict:
                            anc_pos = qpos3CX_hhx+1
                            q3CX = heavyHEX_dict[qpos3CX_hhx]
                            anc = heavyHEX_dict[anc_pos]
                            self.LS_ancillas.append(anc)
                            self.LS_anc_pos.append(rot_qpos(anc_pos))
                            self.LSZZ_pos.append(rot_qpos(anc_pos))
                            self.noisy_cx(qc,[q3CX,anc])
                            self.skip_CX(qc,[qposBS_hhx],None,[anc_pos],None,version='a')
                        else:
                            anc_pos = qpos3CX_hhx+1
                            q3CX = heavyHEX_dict[qpos3CX_hhx]
                            self.LS_ancillas.append(q3CX)
                            self.LS_anc_pos.append(rot_qpos(anc_pos))
                            self.LSZZ_pos.append(rot_qpos(anc_pos))
                            self.noisy_cx(qc,[heavyHEX_dict[qposBS_hhx],q3CX])
                            fake_anc_CX = [heavyHEX_dict[qposBS_hhx],q3CX]
                if 'X' in log_gate[log_round.index(T)]:
                    for qpos3CX,qposBS in zip(edgeX_qubit_pos3CX,edgeX_qubit_posBS):
                        qpos3CX_hhx,qposBS_hhx = rot_qpos(qpos3CX,-1),rot_qpos(qposBS,-1)
                        anc_pos = qpos3CX_hhx-1
                        q3CX = heavyHEX_dict[qpos3CX_hhx]
                        anc = heavyHEX_dict[anc_pos]
                        self.LS_ancillas.append(anc)
                        self.LS_anc_pos.append(rot_qpos(anc_pos))
                        self.LSXX_pos.append(rot_qpos(anc_pos))
                        self.noisy_h(qc,[anc])
                        self.noisy_cx(qc,[anc,q3CX])
                        self.skip_CX(qc,[anc_pos],None,[qposBS_hhx],None,version='a')
                        self.noisy_h(qc,[anc])
                if 'XY' in log_gate[log_round.index(T)]:
                    self.LSYY_pos = self.LSXX_pos[1:]
                    self.LSZZ_pos = self.LSXX_pos[:1]
                    for qpos3CX,qposBS in zip(edgeZ_qubit_pos3CX,edgeZ_qubit_posBS):
                        qpos3CX_hhx,qposBS_hhx = rot_qpos(qpos3CX,-1),rot_qpos(qposBS,-1)
                        anc_pos = qpos3CX_hhx+1
                        q3CX,qBS = heavyHEX_dict[qpos3CX_hhx],heavyHEX_dict[qposBS_hhx]
                        anc = heavyHEX_dict[anc_pos]
                        self.LS_ancillas.append(anc)
                        self.LS_anc_pos.append(rot_qpos(anc_pos))
                        self.LSYY_pos.append(rot_qpos(anc_pos))
                        self.LSZZ_pos.append(rot_qpos(anc_pos))
                        if qpos3CX in edgeX_qubit_pos3CX:
                            self.noisy_s(qc,[q3CX,qBS], dag=True)
                            self.noisy_h(qc,[q3CX,qBS])
                        self.noisy_cx(qc,[q3CX,anc])
                        self.skip_CX(qc,[qposBS_hhx],None,[anc_pos],None,version='a')
                        if qpos3CX in edgeX_qubit_pos3CX:
                            self.noisy_h(qc,[q3CX,qBS])
                            self.noisy_s(qc,[q3CX,qBS])
                if 'XZ' in log_gate[log_round.index(T)]:
                    self.LSYY_pos = self.LSXX_pos + self.LSZZ_pos
                if fake_anc_CX==-1:
                    ##we neeed to finish with a Bell measurement
                    for q1,q2 in zip(qubit_index_list3CX,qubit_index_listBS):
                        self.noisy_cx(qc,[q1,q2])


        if self.LS_ancillas != []:
            self.noisy_measure(qc,self.LS_ancillas,[num_meas_per_round*T + i for i in range(len(self.LS_ancillas))])
            measuredict.update({
                (self.LS_anc_pos[i],T): num_meas_per_round*T + i
                for i in range(len(self.LS_ancillas))
                })
        if fake_anc_CX!=-1:
            self.noisy_cx(qc,fake_anc_CX)
            ##we neeed to finish with a Bell measurement
            for q1,q2 in zip(qubit_index_list3CX,qubit_index_listBS):
                self.noisy_cx(qc,[q1,q2])

        #final measurements
        if logical_meas[0] == 'X':
            self.noisy_h(qc,qubit_index_list3CX)
        if logical_meas[1] == 'X':
            self.noisy_h(qc,qubit_index_listBS)
        self.noisy_measure(qc,qubit_index_list3CX,[num_meas_per_round*T + len(self.LS_ancillas) + i for i in range(d**2)])
        measuredict.update({
            (qubit_pos3CX[i],T): num_meas_per_round*T + len(self.LS_ancillas) + i 
            for i in range(d**2)
            })
        self.noisy_measure(qc,qubit_index_listBS,[num_meas_per_round*T + len(self.LS_ancillas) + d**2 + i for i in range(d**2)])
        measuredict.update({
            (qubit_posBS[i],T): num_meas_per_round*T + len(self.LS_ancillas) + d**2 + i 
            for i in range(d**2)
            })

        self.circuit = qc
        self.measuredict = measuredict
        if log_gate != ['']*len(log_round):
            self.error_sensitive_events = self.get_CX_error_sensitive_events()
        elif len(CX_schedule)==8:
            self.error_sensitive_events = self.get_error_sensitive_events()
        elif '3CX0' not in CX_schedule:
            self.error_sensitive_events = self.get_error_sensitive_events(onlyBS=True)
        elif 'BSX0' not in CX_schedule:
            self.error_sensitive_events = self.get_error_sensitive_events(only3CX=True)


    def noisy_reset(self,qc,q_list):
        # use self.noisy_reset(qc,q_list) instead of qc.reset(q_list)
        qc.reset(q_list)
        if self.Rerror != [0,0]:
            for q in q_list:
                if q in self.vertex_qubits:
                    qc.append(pauli_error([("I",1-self.Rerror[0]),("X",self.Rerror[0])]),[q])
                else:
                    qc.append(pauli_error([("I",1-self.Rerror[1]),("X",self.Rerror[1])]),[q])
        pass

    def noisy_measure(self,qc,q_list,c_list):
        # use self.noisy_measure(qc,q_list,clist) instead of qc.measure(q_list,c_list)
        if self.Rerror != [0,0]:
            for q in q_list:
                if q in self.vertex_qubits:
                    qc.append(pauli_error([("I",1-self.Rerror[0]),("X",self.Rerror[0])]),[q])
                else:
                    qc.append(pauli_error([("I",1-self.Rerror[1]),("X",self.Rerror[1])]),[q])
        qc.measure(q_list,c_list)
        pass

    def noisy_h(self,qc,q_list):
        # use self.noisy_h(qc,q_list) instead of qc.h(q_list)
        qc.h(q_list)
        if self.singleQerror > 0:
            for q in q_list:
                qc.append(depolarizing_error(self.singleQerror,1),[q])
        pass
    
    def noisy_s(self,qc,q_list, dag = False):
        # use self.noisy_s(qc,q_list, dag=True) instead of qc.sdg(q_list)
        if dag:
            qc.sdg(q_list)
        else:
            qc.s(q_list)
        if self.singleQerror > 0:
            for q in q_list:
                qc.append(depolarizing_error(self.singleQerror,1),[q])
        pass

    def noisy_cx(self,qc,qubits):
        # use self.noisy_cx(qc,[control_qubit, target_qubit]) instead of qc.cx([control_qubit, target_qubit])
        qc.cx(qubits[0],qubits[1])
        if self.CXerror>0:
            qc.append(depolarizing_error(self.CXerror, 2), qubits)
        pass

    def skip_CX(self, qc,qa_pos_list,qb_pos_list,qc_pos_list,qd_pos_list, version = 'a'):
        """a->c (through b); b->d (through c). """
        if qb_pos_list==qd_pos_list==None:
            for qa_pos,qb_pos in zip(qa_pos_list,qc_pos_list):
                qa = self.heavyHEX_dict[qa_pos]
                qb = self.heavyHEX_dict[qb_pos]
                if not self.virt_link:
                    if (qa_pos+qb_pos)/2 in self.heavyHEX_dict:
                        qm = self.heavyHEX_dict[(qa_pos+qb_pos)/2]
                    elif (qa_pos+qb_pos)/2 not in self.heavyHEX_dict:
                        qm_pos = [(qa_pos+qb_pos+c)/2 for c in [1+1j,1-1j,-1+1j,-1-1j] 
                                  if (qa_pos+qb_pos+c)/2 in self.heavyHEX_dict and
                                  (qa_pos+qb_pos+c)/2 not in [qa_pos,qb_pos]
                                  ][0]
                        qm = self.heavyHEX_dict[qm_pos]

                    if version == 'a':
                        self.noisy_cx(qc,[qm,qb])
                        self.noisy_cx(qc,[qa,qm])
                        self.noisy_cx(qc,[qm,qb])
                        self.noisy_cx(qc,[qa,qm])
                    elif version == 'b':
                        self.noisy_cx(qc,[qa,qm])
                        self.noisy_cx(qc,[qm,qb])
                        self.noisy_cx(qc,[qa,qm])
                        self.noisy_cx(qc,[qm,qb])

                else:
                    self.noisy_cx(qc,[qa,qb])

        else:
            for qa_pos,qb_pos,qc_pos,qd_pos in zip(qa_pos_list,qb_pos_list,qc_pos_list,qd_pos_list):
                q_a = self.heavyHEX_dict[qa_pos]
                q_b = self.heavyHEX_dict[qb_pos]
                q_c = self.heavyHEX_dict[qc_pos]
                q_d = self.heavyHEX_dict[qd_pos]

                #start with the first control in the straightened coupling map            
                if q_a+1 == q_b or q_a-1 == q_b:
                    q_a,q_b = q_b,q_a
                    q_c,q_d = q_d,q_c
        
                if not self.virt_link:
                    self.noisy_cx(qc,[q_a,q_b])
                    self.noisy_cx(qc,[q_b,q_c])
                    self.noisy_cx(qc,[q_a,q_b])
                    self.noisy_cx(qc,[q_c,q_d])
                    self.noisy_cx(qc,[q_b,q_c])
                    self.noisy_cx(qc,[q_c,q_d])

                else:
                    self.noisy_cx(qc,[q_a,q_c])
                    self.noisy_cx(qc,[q_b,q_d])
        pass

    def buildHeavyHEX(self, Nq):
        if Nq == 127:
            rows, cols = 13,15
            HHXlatticepos = [[i,0,i] for i in range(cols-1)]
            i=cols-1

        elif Nq == 433:
            rows, cols = 25,27
            HHXlatticepos = [[i,0,i] for i in range(cols-1)]
            i=cols-1

        elif Nq == 133:
            rows, cols = 14,15
            HHXlatticepos = [[i,0,i] for i in range(cols)]
            i=cols

        row = 1
        while row < rows-1:
            col=0
            if row%2==0:
                while col < cols:
                    HHXlatticepos.append([i, row, col])
                    i+=1
                    col+=1
            else:
                while col < cols:
                    if row%4==1 and col%4==0:
                        HHXlatticepos.append([i, row, col])
                        i+=1
                    if row%4==3 and col%4==2:
                        HHXlatticepos.append([i, row, col])
                        i+=1
                    col+=1
            row+=1
        if Nq != 133:    
            HHXlatticepos.extend([[len(HHXlatticepos)+i,rows-1,i+1] for i in range(cols-1)])
        else:
            col=0
            while col < cols:
                if row%4==1 and col%4==0:
                    HHXlatticepos.append([i, row, col])
                    i+=1
                if row%4==3 and col%4==2:
                    HHXlatticepos.append([i, row, col])
                    i+=1
                col+=1
                
        heavyHEX_dict = {qr+qi*1j: i for i,qi,qr in HHXlatticepos}
        return heavyHEX_dict

    def sorted_complex(self, xs):
        return sorted(xs, key=lambda v: (v.real+v.imag, v.imag-v.real))
    
    def draw_lattice(self, indices:bool = True):
        qi_list, y_list, x_list = np.transpose([[item[1],item[0].imag,item[0].real] for item in self.heavyHEX_dict.items()])
    
        yq3_list, xq3_list = np.transpose([[item[0].imag,item[0].real] for item in self.q2i3CX.items()])
        ya3_list, xa3_list = np.transpose([[item[0].imag,item[0].real] for item in self.a2i3CX.items()])

        yq_list, xq_list = np.transpose([[item[0].imag,item[0].real] for item in self.q2iBS.items()])
        ya_list, xa_list = np.transpose([[item[0].imag,item[0].real] for item in self.a2iBS.items()])

        plt.figure(frameon=False)
        plt.plot(x_list,-y_list,'o',c='gray',alpha = 0.65)
        for i in [int(qi) for qi in qi_list]:
            if indices:
                plt.text(x_list[i],-y_list[i],str(i),fontsize=7)

        plt.plot(xq3_list,-yq3_list,'o',c='b')
        plt.plot(xa3_list,-ya3_list,'o',c='r')
        plt.plot(xq_list,-yq_list,'o',c='c')
        plt.plot(xa_list,-ya_list,'o',c='m')
        plt.show()
        pass

    def get_error_sensitive_events(self, return_value = 'meas_inds', every_initial_stab = False, only3CX = False, onlyBS = False):
        d = self.d
        T = self.T
        # qubit_pos = self.qubit_pos
        blue0_rows = [[re+im*1j for re in range(4,4*d,8)]+[4*d+im*1j] for im in range(8,4*d,8)] #blue if it starts with a complete blue plaquette at meas cycle 0
        red0_rows = [[re+im*1j for re in range(8,4*d,8)]+[4*d+im*1j] for im in range(4,4*d,8)]
        blue0_cols = [[re+im*1j for im in range(8,4*d,8)]+[re+4*d*1j] for re in range(8,4*d,8)]
        red0_cols = [[re+im*1j for im in range(4,4*d,8)]+[re+4*d*1j] for re in range(4,4*d,8)]

        blue1_rows = [[re+im*1j for re in range(4,4*d,8)]+[4*d+im*1j] for im in range(4,4*d,8)] #blue if it starts with a complete blue plaquette at meas cycle 1
        red1_rows = [[re+im*1j for re in range(8,4*d,8)]+[4*d+im*1j] for im in range(8,4*d,8)]
        blue1_cols = [[re+im*1j for im in range(8,4*d,8)]+[re+4*d*1j] for re in range(4,4*d,8)]
        red1_cols = [[re+im*1j for im in range(4,4*d,8)]+[re+4*d*1j] for re in range(8,4*d,8)]

        detectors_coord = []
        ### detectors for the 3CX code
        if not onlyBS:
            if d%2==1:
                for time in range(T):
                    if time!=0:
                        if time%2==1:
                            bluet_rows = blue0_rows
                            redt_rows = red0_rows
                            bluet_cols = blue0_cols
                            redt_cols = red0_cols
                        elif time%2==0:
                            bluet_rows = blue1_rows
                            redt_rows = red1_rows
                            bluet_cols = blue1_cols
                            redt_cols = red1_cols

                        for blue_row in bluet_rows:
                            detectors_coord.append([(blue_row[0],time)])
                            for blue_pos in blue_row[1:-1]:
                                detectors_coord.append([(blue_pos-4,time-1),(blue_pos,time)]) 
                            detectors_coord.append([(blue_row[-1]-4,time-1),(blue_row[-1],time-1),(blue_row[-1],time)])
                        for red_row in redt_rows:
                            for blue_pos in red_row[:-1]:
                                detectors_coord.append([(blue_pos-4,time-1),(blue_pos,time)])
                            detectors_coord.append([(red_row[-1],time-1),(red_row[-1],time)])
                        for blue_col in bluet_cols:
                            for red_pos in blue_col[:-1]:
                                detectors_coord.append([(red_pos-4j,time-1),(red_pos,time)])
                            detectors_coord.append([(blue_col[-1],time-1),(blue_col[-1],time)])
                        for red_col in redt_cols:
                            detectors_coord.append([(red_col[0],time)])
                            for red_pos in red_col[1:-1]:
                                detectors_coord.append([(red_pos-4j,time-1),(red_pos,time)])
                            detectors_coord.append([(red_col[-1]-4j,time-1),(red_col[-1],time-1),(red_col[-1],time)])
                    
                    if time==0 and (self.logical_prep[0]=='Z' or every_initial_stab):
                        for blue_row in blue1_rows:
                            detectors_coord.append([(blue_row[0],time)])
                            for blue_pos in blue_row[1:-1]:
                                detectors_coord.append([(blue_pos,time)])
                            detectors_coord.append([(blue_row[-1],time)])
                        for red_row in red1_rows:
                            for blue_pos in red_row[:-1]:
                                detectors_coord.append([(blue_pos,time)])
                            detectors_coord.append([(red_row[-1],time)])
                    if time==0 and (self.logical_prep[0]=='X' or every_initial_stab):
                        for blue_col in blue1_cols:
                            for red_pos in blue_col[:-1]:
                                detectors_coord.append([(red_pos,time)])
                            detectors_coord.append([(blue_col[-1],time)])
                        for red_col in red1_cols:
                            detectors_coord.append([(red_col[0],time)])
                            for red_pos in red_col[1:-1]:
                                detectors_coord.append([(red_pos,time)])
                            detectors_coord.append([(red_col[-1],time)])
                #final measurement detectors
                if self.logical_meas[0] == 'Z':
                    if T%2==0:
                        bluet_rows = blue0_rows
                        redt_rows = red0_rows
                    elif T%2==1:
                        bluet_rows = blue1_rows
                        redt_rows = red1_rows
                    for blue_row in bluet_rows:
                        for blue_pos in blue_row[:-1]:
                            detector = []
                            detector.append((blue_pos,T-1))
                            for rel_pos in [2+2j,2-2j,-2+2j,-2-2j]:
                                detector.append((blue_pos+rel_pos,T))
                            detectors_coord.append(detector)
                        detectors_coord.append([(blue_row[-1],T-1),(blue_row[-1]-2-2j,T),(blue_row[-1]-2+2j,T)])
                    for red_row in redt_rows:
                        detectors_coord.append([(red_row[0]-6-2j,T),(red_row[0]-6+2j,T)])
                        for blue_pos in red_row[:-1]:
                            detector = []
                            detector.append((blue_pos,T-1))
                            if blue_pos==red_row[-2]:
                                detector.append((blue_pos+4,T-1))
                            for rel_pos in [2+2j,2-2j,-2+2j,-2-2j]:
                                detector.append((blue_pos+rel_pos,T))
                            detectors_coord.append(detector)
                elif self.logical_meas[0] == 'X':
                    if T%2==0:
                        bluet_cols = blue0_cols
                        redt_cols = red0_cols
                    elif T%2==1:
                        bluet_cols = blue1_cols
                        redt_cols = red1_cols
                    for blue_col in bluet_cols:
                        detectors_coord.append([(blue_col[0]-2-6j,T),(blue_col[0]+2-6j,T)])
                        for red_pos in blue_col[:-1]:
                            detector = []
                            detector.append((red_pos,T-1))
                            if red_pos == blue_col[-2]:
                                detector.append((red_pos+4j,T-1))
                            for rel_pos in [2+2j,2-2j,-2+2j,-2-2j]:
                                detector.append((red_pos+rel_pos,T))
                            detectors_coord.append(detector)
                    for red_col in redt_cols:
                        for red_pos in red_col[:-1]:
                            detector = []
                            detector.append((red_pos,T-1))
                            for rel_pos in [2+2j,2-2j,-2+2j,-2-2j]:
                                detector.append((red_pos+rel_pos,T))
                            detectors_coord.append(detector)
                        detectors_coord.append([(red_col[-1],T-1),(red_col[-1]-2-2j,T),(red_col[-1]+2-2j,T)])

            elif d%2==0:
                for time in range(T):
                    if time!=0:
                        if time%2==1:
                            bluet_rows = blue0_rows
                            redt_rows = red0_rows
                            bluet_cols = blue0_cols
                            redt_cols = red0_cols
                        elif time%2==0:
                            bluet_rows = blue1_rows
                            redt_rows = red1_rows
                            bluet_cols = blue1_cols
                            redt_cols = red1_cols
                        for blue_row in bluet_rows:
                            detectors_coord.append([(blue_row[0],time)])
                            for blue_pos in blue_row[1:-1]:
                                detectors_coord.append([(blue_pos-4,time-1),(blue_pos,time)]) 
                            detectors_coord.append([(blue_row[-1],time-1),(blue_row[-1],time)])
                        for red_row in redt_rows:
                            for blue_pos in red_row[:-1]:
                                detectors_coord.append([(blue_pos-4,time-1),(blue_pos,time)])
                            detectors_coord.append([(red_row[-1]-4,time-1),(red_row[-1],time-1),(red_row[-1],time)])
                        for blue_col in bluet_cols:
                            for red_pos in blue_col[:-1]:
                                detectors_coord.append([(red_pos-4j,time-1),(red_pos,time)])
                            detectors_coord.append([(blue_col[-1]-4j,time-1),(blue_col[-1],time-1),(blue_col[-1],time)])
                        for red_col in redt_cols:
                            detectors_coord.append([(red_col[0],time)])
                            for red_pos in red_col[1:-1]:
                                detectors_coord.append([(red_pos-4j,time-1),(red_pos,time)])
                            detectors_coord.append([(red_col[-1],time-1),(red_col[-1],time)])

                    if time==0 and (self.logical_prep[0]=='Z' or every_initial_stab):
                        for blue_row in blue1_rows:
                            for blue_pos in blue_row:
                                detectors_coord.append([(blue_pos,0)]) 
                        for red_row in red1_rows:
                            for blue_pos in red_row:
                                detectors_coord.append([(blue_pos,0)]) 
                    if time==0 and (self.logical_prep[0]=='X' or every_initial_stab):
                        for blue_col in blue1_cols:
                            for red_pos in blue_col:
                                detectors_coord.append([(red_pos,0)]) 
                        for red_col in red1_cols:
                            for red_pos in red_col:
                                detectors_coord.append([(red_pos,0)]) 
                #final measurement detectors
                if self.logical_meas[0] == 'Z':
                    if T%2==0:
                        bluet_rows = blue0_rows
                        redt_rows = red0_rows
                    elif T%2==1:
                        bluet_rows = blue1_rows
                        redt_rows = red1_rows
                    for blue_row in bluet_rows:
                        for blue_pos in blue_row[:-1]:
                            detector = []
                            detector.append((blue_pos,T-1))
                            if blue_pos==blue_row[-2]:
                                detector.append((blue_pos+4,T-1))
                            for rel_pos in [2+2j,2-2j,-2+2j,-2-2j]:
                                detector.append((blue_pos+rel_pos,T))
                            detectors_coord.append(detector)
                    for red_row in redt_rows:
                        detectors_coord.append([(red_row[0]-6-2j,T),(red_row[0]-6+2j,T)])
                        for blue_pos in red_row[:-1]:
                            detector = []
                            detector.append((blue_pos,T-1))
                            for rel_pos in [2+2j,2-2j,-2+2j,-2-2j]:
                                detector.append((blue_pos+rel_pos,T))
                            detectors_coord.append(detector)
                        detectors_coord.append([(red_row[-1],T-1),(red_row[-1]-2-2j,T),(red_row[-1]-2+2j,T)])
                elif self.logical_meas[0] == 'X':
                    if T%2==0:
                        bluet_cols = blue0_cols
                        redt_cols = red0_cols
                    elif T%2==1:
                        bluet_cols = blue1_cols
                        redt_cols = red1_cols
                    for blue_col in bluet_cols:
                        detectors_coord.append([(blue_col[0]-2-6j,T),(blue_col[0]+2-6j,T)])
                        for red_pos in blue_col[:-1]:
                            detector = []
                            detector.append((red_pos,T-1))
                            for rel_pos in [2+2j,2-2j,-2+2j,-2-2j]:
                                detector.append((red_pos+rel_pos,T))
                            detectors_coord.append(detector)
                        detectors_coord.append([(blue_col[-1],T-1),(blue_col[-1]-2-2j,T),(blue_col[-1]+2-2j,T)])
                    for red_col in redt_cols:
                        for red_pos in red_col[:-1]:
                            detector = []
                            detector.append((red_pos,T-1))
                            if red_pos==red_col[-2]:
                                detector.append((red_pos+4j,T-1))
                            for rel_pos in [2+2j,2-2j,-2+2j,-2-2j]:
                                detector.append((red_pos+rel_pos,T))
                            detectors_coord.append(detector)

        ### detectors for the Bacon-Shor code
        if not only3CX:
            for time in range(T):
                #X detectors
                for q_r in range(5,5+4*(d-1),4):
                    if time==0 and (self.logical_prep[1]=='X' or every_initial_stab):
                        detector = [(q_r+1j*q_i,time) for q_i in range(3,3+4*d,4)]
                        detectors_coord.append(detector)
                    if time<T-1:
                        detector = [(q_r+1j*q_i,time) for q_i in range(3,3+4*d,4)]
                        detector += [(q_r+1j*q_i,time+1) for q_i in range(3,3+4*d,4)]
                        detectors_coord.append(detector)
                    elif self.logical_meas[1]=='X':
                        detector = [(q_r+1j*q_i,time) for q_i in range(3,3+4*d,4)]
                        detector += [(q_r+q_rel+1j*q_i,T) for q_i in range(3,3+4*d,4) for q_rel in [-2,+2]]
                        detectors_coord.append(detector)
                #Z detectors
                for q_i in range(5,5+4*(d-1),4):
                    if time==0 and (self.logical_prep[1]=='Z' or every_initial_stab):
                        detector = [(q_r+1j*q_i,time) for q_r in range(3,3+4*d,4)]
                        detectors_coord.append(detector)
                    if time<T-1:
                        detector = [(q_r+1j*q_i,time) for q_r in range(3,3+4*d,4)]
                        detector += [(q_r+1j*q_i,time+1) for q_r in range(3,3+4*d,4)]
                        detectors_coord.append(detector)
                    elif self.logical_meas[1]=='Z':
                        detector = [(q_r+1j*q_i,time) for q_r in range(3,3+4*d,4)]
                        detector += [(q_r+1j*(q_i+q_rel),T) for q_r in range(3,3+4*d,4) for q_rel in [-2,+2]]
                        detectors_coord.append(detector)

        detectors = [] #space-time coordinates to measurement indices
        for det_coords in detectors_coord:
            new_detector = []
            for det_coord in det_coords:
                new_detector.append(self.measuredict[det_coord])
            detectors.append(new_detector)
        if return_value == 'meas_inds':
            return detectors
        elif return_value == 'coords':
            return detectors_coord

    def get_CX_error_sensitive_events(self):
        """Exploiting that the 3CX code lives on even (rotated) coordinates, and the Bacon-Shor on odds."""

        if 0 not in self.log_round:
            detectors_coord = self.get_error_sensitive_events(return_value='coords')
        else:
            detectors_coord = self.get_error_sensitive_events(every_initial_stab=True,return_value='coords')

        #detectors not in the logical round are unchanged
        relevant_detectors_coord = []
        for det_coords in detectors_coord:
            _,time = det_coords[-1]
            if time not in self.log_round:
                relevant_detectors_coord.append(det_coords)
            elif self.log_gate[self.log_round.index(time)]=='':
                relevant_detectors_coord.append(det_coords)

        for log_rd in self.log_round:
            Xdet_coords3CX = [] #time of CX
            Zdet_coords3CX = []
            Xdet_coordsBS = [] 
            Zdet_coordsBS = [] 
            for det_coords in detectors_coord:
                meas_coord,time = det_coords[-1]
                if time == log_rd:
                    if meas_coord in self.measure_x_qubit_posBS:
                        Xdet_coordsBS.append(det_coords)
                    elif meas_coord in self.measure_z_qubit_posBS:
                        Zdet_coordsBS.append(det_coords) 
                    elif time%2==1 and meas_coord in self.measure_x_qubit_pos_cycle0:
                        Xdet_coords3CX.append(det_coords)
                    elif time%2==1 and meas_coord in self.measure_z_qubit_pos_cycle0:
                        Zdet_coords3CX.append(det_coords)
                    elif time%2==0 and meas_coord in self.measure_x_qubit_pos_cycle1:
                        Xdet_coords3CX.append(det_coords)
                    elif time%2==0 and meas_coord in self.measure_z_qubit_pos_cycle1:
                        Zdet_coords3CX.append(det_coords)

                    elif time == log_rd == self.T:
                        if meas_coord in self.qubit_pos3CX:
                            if self.logical_meas[0] == 'X':
                                Xdet_coords3CX.append(det_coords)
                            if self.logical_meas[0] == 'Z':
                                Zdet_coords3CX.append(det_coords)
                        if meas_coord in self.qubit_posBS:
                            if self.logical_meas[1] == 'X':
                                Xdet_coordsBS.append(det_coords)
                            if self.logical_meas[1] == 'Z':
                                Zdet_coordsBS.append(det_coords)

            Xdet_coords3CX_merged = []
            for re in range(4,4*self.d,4):
                new_det_coord= []
                for det_coords in Xdet_coords3CX:
                    # meas_coord,_ = det_coords
                    meas_coord = sum([coord for coord,_ in det_coords])/len(det_coords)
                    if meas_coord.real == re:
                        new_det_coord.extend(det_coords)
                Xdet_coords3CX_merged.append(new_det_coord)
            if Xdet_coordsBS == []: #for the zip later
                Xdet_coordsBS = [[]]*len(Xdet_coords3CX_merged)

            Zdet_coords3CX_merged = []
            for im in range(4,4*self.d,4):
                new_det_coord= []
                for det_coords in Zdet_coords3CX:
                    # meas_coord,_ = det_coords[0]
                    meas_coord = sum([coord for coord,_ in det_coords])/len(det_coords)
                    if meas_coord.imag == im:
                        new_det_coord.extend(det_coords)
                Zdet_coords3CX_merged.append(new_det_coord)
            if Zdet_coordsBS == []:
                Zdet_coordsBS = [[]]*len(Zdet_coords3CX_merged)
    
            #append merged detectors for the logical round
            if self.log_gate[self.log_round.index(log_rd)] in ['CX','LSXZ','LSXY']:
                for det_coords in Zdet_coords3CX:
                    if log_rd!=0 or (log_rd==self.T and self.logical_meas==['Z']*2):
                        #detectors unaffected by the transversal CX
                        relevant_detectors_coord.append(det_coords)
                for det_coords in Xdet_coordsBS:
                    if log_rd!=0 or (log_rd==self.T and self.logical_meas==['X']*2):
                        #detectors unaffected by the transversal CX
                        relevant_detectors_coord.append(det_coords)
                for det_coords3CX,det_coords2 in zip(Xdet_coords3CX_merged,Xdet_coordsBS):
                    if log_rd<self.T:
                        #X detectors from 3CX and BS merge on the 3CX (control) side
                        det_coordsBS = []
                        for coords in det_coords2:
                            if coords[1]==log_rd:
                                det_coordsBS.append(coords)
                        relevant_detectors_coord.append(det_coords3CX+det_coordsBS)                    
                    elif self.logical_meas!=['Z']*2:
                        #X detectors from 3CX (control) side propagate to both codes                    
                        avg_coord = sum([coord for coord,_ in det_coords3CX])/len(det_coords3CX)
                        det_coordsBS = [(avg_coord.real+1j*im + 1 -1j,log_rd-1) for im in range(4,4*self.d+1,4)]
                        relevant_detectors_coord.append(det_coords3CX+det_coordsBS)
                for det_coords1,det_coordsBS in zip(Zdet_coords3CX_merged,Zdet_coordsBS):
                    if log_rd<self.T:
                        #Z detectors from 3CX and BS merge on the BS (target) side
                        det_coords3CX = []
                        for coords in det_coords1:
                            if coords[1]==log_rd:
                                det_coords3CX.append(coords)                        
                        relevant_detectors_coord.append(det_coords3CX+det_coordsBS)                    
                    elif self.logical_meas!=['X']*2:
                        #Z detectors from BS (target) side propagate to both codes
                        avg_coord = sum([coord for coord,_ in det_coordsBS])/len(det_coordsBS)
                        if self.T%2==1:
                            Zstab_coords3CX = self.measure_z_qubit_pos_cycle1
                        else:
                            Zstab_coords3CX = self.measure_z_qubit_pos_cycle0
                        det_coords3CX = [(pos,log_rd-1) for pos in Zstab_coords3CX if pos.imag==avg_coord.imag-1]
                        relevant_detectors_coord.append(det_coords3CX+det_coordsBS)
            
            if self.log_gate[self.log_round.index(log_rd)] == 'XC':
                for det_coords in Xdet_coords3CX: 
                    if log_rd!=0 or (log_rd==self.T and self.logical_meas==['X']*2):
                        relevant_detectors_coord.append(det_coords)
                for det_coords in Zdet_coordsBS:
                    if log_rd!=0 or (log_rd==self.T and self.logical_meas==['Z']*2):
                        relevant_detectors_coord.append(det_coords)
                for det_coords3CX,det_coords2 in zip(Zdet_coords3CX_merged,Zdet_coordsBS):
                    if log_rd<self.T:
                        det_coordsBS = []
                        for coords in det_coords2:
                            if coords[1]==log_rd:
                                det_coordsBS.append(coords)                        
                        relevant_detectors_coord.append(det_coords3CX+det_coordsBS)                                        
                    elif self.logical_meas!=['X']*2:
                        avg_coord = sum([coord for coord,_ in det_coords3CX])/len(det_coords3CX)
                        det_coordsBS = [(1j*avg_coord.imag + re - 1 + 1j,log_rd-1) for re in range(4,4*self.d+1,4)]
                        print(det_coordsBS)
                        relevant_detectors_coord.append(det_coords3CX+det_coordsBS)
                for det_coords1,det_coordsBS in zip(Xdet_coords3CX_merged,Xdet_coordsBS):
                    if log_rd<self.T:
                        det_coords3CX = []
                        for coords in det_coords1:
                            if coords[1]==log_rd:
                                det_coords3CX.append(coords)                        
                        relevant_detectors_coord.append(det_coords3CX+det_coordsBS)                    
                    elif self.logical_meas!=['Z']*2:
                        avg_coord = sum([coord for coord,_ in det_coordsBS])/len(det_coordsBS)
                        if self.T%2==1:
                            Xstab_coords3CX = self.measure_x_qubit_pos_cycle1
                        else:
                            Xstab_coords3CX = self.measure_x_qubit_pos_cycle0
                        det_coords3CX = [(pos,log_rd-1) for pos in Xstab_coords3CX if pos.real==avg_coord.real-1]
                        relevant_detectors_coord.append(det_coords3CX+det_coordsBS)


        detectors = [] #space-time coordinates to measurement indices
        for det_coords in relevant_detectors_coord:
            new_detector = []
            for det_coord in det_coords:
                new_detector.append(self.measuredict[det_coord])
            if new_detector != []:
                detectors.append(new_detector)

        return detectors