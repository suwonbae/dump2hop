import numpy as np
import time
from mpi4py import MPI
import matplotlib.pyplot as plt

def d_pbc(vector1,vector2,boxlength):
    l_ref=[boxlength[0]/2.0,boxlength[1]/2.0]
    vector=vector1-vector2

    if vector[0]<(-1)*l_ref[0]:
        vector[0]=vector[0]+boxlength[0]
    elif vector[0]>l_ref[0]:
        vector[0]=vector[0]-boxlength[0]
    else:
        vector[0]=vector[0]

    if vector[1]<(-1)*l_ref[1]:
        vector[1]=vector[1]+boxlength[1]
    elif vector[1]>l_ref[1]:
        vector[1]=vector[1]-boxlength[1]
    else:
        vector[1]=vector[1]

    return vector

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

mol_id = 100 #chain corresponding to mol_id to be visualized
N = 48000 #number of beads/atoms of interest (from atom 1 to atom N / atom 0 to atom N-1)
subs = 0 #1 if for subsequent simulation
filename = 'further6.png'
f_path = '/data/bcpfilm/pure/N20_f25_48000/equil/1.2/further/further2/further3/further4/further5/further6/'
f_name = 'dump.'
f_ind = np.linspace(0, 10000000, 1001, dtype=int) #file index

avg_rows_per_process = int(len(f_ind)/size)

if rank == 0:
    t0 = time.time()

    box = []

    infile = f_path + '/' + f_name + '0'
    f = open(infile, 'r')
    for iind in range(5):
        f.readline()
    for iind in range(3):
        line = f.readline().split()
        box.append([float(line[0]), float(line[1])])
    f.close()

    trj_init = np.loadtxt(infile, skiprows=9)
    trj_init = trj_init[np.lexsort(np.fliplr(trj_init).T)][:N,:]

    #molecule of interest at the starting point
    moi_init = trj_init[trj_init[:,1] == mol_id,:]
 
else:
    box = None
    moi_init = None

box = comm.bcast(box, root=0)
moi_init = comm.bcast(moi_init, root=0)

dis = {}

start_row = rank * avg_rows_per_process
end_row = start_row + avg_rows_per_process

end_ind = end_row + 1
if rank == size-1:
    end_row = len(f_ind)
    end_ind = end_row

for iind in range(start_row, end_ind):
    infile = f_path + '/' + f_name + str(f_ind[iind])
    trj = np.loadtxt(infile, skiprows=9)
    trj = trj[np.lexsort(np.fliplr(trj).T)][:N,:]

    #molecules of interest at given timestep
    moi = trj[trj[:,1] == mol_id,:]
        
    dis_temp = np.zeros(moi.shape)
    dis_temp[:,:3] = moi[:,:3]

    if iind > start_row: #start_row for rank of 0 is equal to 0
        for jind in range(moi.shape[0]):
            #displacement vector over pbc
            dis_temp[jind,3:] = d_pbc(moi[jind,3:],moi_prev[jind,3:],[box[0][1]-box[0][0], box[1][1]-box[1][0]])

        dis_temp[:,3:] += dis[iind-1][:,3:]

    dis.update({iind : dis_temp})
    moi_prev = moi

'calculate displacements'
if rank == 0:
    #even though rank 0 processor processes from dis[0=start_row] to dis[end_row-1],
    #dis[end_row] works as a bridge b/w ranks 0 and 1, and has to be sent to rank 1
    data = dis[end_row] 
    if size > 1:
        req = comm.Isend(data, dest=(rank+1))
        req.Wait()

elif rank == size-1:
    data = np.empty(moi.shape)
    req = comm.Irecv(data, source=(rank-1))
    req.Wait()

    for iind in range(start_row, end_row):
        dis[iind][:,3:] += data[:,3:]

else:
    data = np.empty(moi.shape)
    req = comm.Irecv(data, source=(rank-1))
    req.Wait()

    #even though rank n processor processes from dis[start_row] to dis[end_row-1],
    #dis[end_row] works as a bridge b/w ranks n and n+1, and has to be sent to rank n+1
    for iind in range(start_row, end_row+1): 
        dis[iind][:,3:] += data[:,3:]

    data = dis[end_row] #dis[end_row] works as a bridge b/w adjacent processors
    req = comm.Isend(data, dest=(rank+1))
    req.Wait()

'calculates and collect msd from all procs'
if rank == 0:
    t1 = time.time() - t0
    print (t1)

    centers = np.zeros([end_row - start_row,2])
    for iind in range(start_row, end_row):
        centers[iind - start_row,:] = np.average((moi_init + dis[iind])[:2,3:5],axis=0)

    for iind in range(1, size):
        start_row = iind * avg_rows_per_process
        end_row = start_row + avg_rows_per_process
        if iind == size-1:
            end_row = len(f_ind)

        centers_temp = np.empty([end_row - start_row,2])
        req = comm.Irecv(centers_temp, source=iind)
        req.Wait()

        centers = np.vstack((centers, centers_temp))

else:
    centers_temp = np.zeros([end_row - start_row,2])
    for iind in range(start_row, end_row):
        centers_temp[iind - start_row,:] = np.average((moi_init + dis[iind])[:2,3:5],axis=0)

    req = comm.Isend(centers_temp, dest=0)
    req.Wait()

'save msd'
if rank == 0:
    t2 = time.time() - t0
    print (t2)

    #calculate traveling distance until arriving at the endig point instead of counthing the number of hopping events
    segment = 0
    for iind in range(1,len(centers)):
        segment += np.linalg.norm(centers[iind] - centers[iind-1])
    print (segment)

    #color varies with timestep
    plt.figure()
    bins = 10
    for iind in range(bins):
        color = [1.0/(bins+1)*(iind+1),1.0/(bins+1)*(iind+1),1.0/(bins+1)*(iind+1)]
        plt.plot(centers[len(centers)//bins*(iind):len(centers)//bins*(iind+1)+1,0], centers[len(centers)//bins*(iind):len(centers)//bins*(iind+1)+1,1], color=color, lw=0.5)

    plt.plot(centers[0,0], centers[0,1], 'ko', ms=3, label='starting point')
    plt.plot(centers[-1,0], centers[-1,1], 'ks', ms=3, label='ending point')
    plt.xlim(-57,57)
    plt.ylim(0,90)
    plt.gca().set_aspect('equal',adjustable='box')
    plt.savefig(filename, dpi=300)
