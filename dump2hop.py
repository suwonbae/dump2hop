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

mol_id = 1 #chain corresponding to mol_id to be visualized
N = 48000 #number of only beads/atoms constructing chains (from atom 1 to atom N / atom 0 to atom N-1)
n = 2400 #number of chains/molecules
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

else:
    box = None
    trj_init = None

box = comm.bcast(box, root=0)
trj_init = comm.bcast(trj_init, root=0)

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

    dis_temp = np.zeros(trj.shape)
    dis_temp[:,:3] = trj[:,:3]

    if iind > start_row: #start_row for rank of 0 is equal to 0
        for jind in range(trj.shape[0]):
            #displacement vector over pbc
            dis_temp[jind,3:] = d_pbc(trj[jind,3:],trj_prev[jind,3:],[box[0][1]-box[0][0], box[1][1]-box[1][0]])

        dis_temp[:,3:] += dis[iind-1][:,3:]

    dis.update({iind : dis_temp})
    trj_prev = trj

'calculate displacements'
if rank == 0:
    #even though rank 0 processor processes from dis[0=start_row] to dis[end_row-1],
    #dis[end_row] works as a bridge b/w ranks 0 and 1, and has to be sent to rank 1
    data = dis[end_row] 
    if size > 1:
        req = comm.Isend(data, dest=(rank+1))
        req.Wait()

elif rank == size-1:
    data = np.empty(trj.shape)
    req = comm.Irecv(data, source=(rank-1))
    req.Wait()

    for iind in range(start_row, end_row):
        dis[iind][:,3:] += data[:,3:]

else:
    data = np.empty(trj.shape)
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

    centers = np.zeros([end_row - start_row,2,n])
    for iind in range(start_row, end_row):
        trj = np.zeros(trj_init.shape)
        trj[:,:3] = trj_init[:,:3]
        trj[:,3:] = trj_init[:,3:] + dis[iind][:,3:]
        for jind in range(n):
            logic = trj[:,1] == (jind+1)
            moi = trj[logic]
            centers[iind - start_row,:,jind] = np.average(moi[:2,3:5],axis=0)

    for iind in range(1, size):
        start_row = iind * avg_rows_per_process
        end_row = start_row + avg_rows_per_process
        if iind == size-1:
            end_row = len(f_ind)

        centers_temp = np.empty([end_row - start_row,2,n])
        req = comm.Irecv(centers_temp, source=iind)
        req.Wait()

        centers = np.vstack((centers, centers_temp)) #shape of final corresponds to (timesteps, 2, n)

else:
    centers_temp = np.zeros([end_row - start_row,2,n])
    for iind in range(start_row, end_row):
        trj = np.zeros(trj_init.shape)
        trj[:,:3] = trj_init[:,:3]
        trj[:,3:] = trj_init[:,3:] + dis[iind][:,3:]
        for jind in range(n):
            logic = trj[:,1] == (jind+1)
            moi = trj[logic]
            centers_temp[iind - start_row,:,jind] = np.average(moi[:2,3:5],axis=0)

    req = comm.Isend(centers_temp, dest=0)
    req.Wait()

'save msd'
if rank == 0:
    t2 = time.time() - t0
    print (t2)

    #calculate traveling distance until arriving at the endig point instead of counthing the number of hopping events
    segments = []
    for iind in range(n):
        segment = 0
        for jind in range(1,len(centers)):
            segment += np.linalg.norm(centers[jind,:,iind] - centers[jind-1,:,iind])
        segments.append(segment)
    
    #mol_id's which exhibits min and max among n segment values
    min_id = np.argmin(segments)
    max_id = np.argmax(segments)

    t3 = time.time() - t0
    print (t3)

    plt.figure()
    bins = np.linspace(1600,2400,17)
    hist, bin_edges = np.histogram(segments, bins=bins, density=True)
    plt.plot(bin_edges[1:], hist)
    plt.xlabel(r'travel distance ($\sigma$)')
    plt.ylabel('probability')
    plt.savefig('segments.png',dpi=300)

    plt.figure()
    plt.plot(centers[:,0,max_id], centers[:,1,max_id], lw=0.5)
    plt.plot(centers[0,0,max_id], centers[0,1,max_id], 'ko', ms=3, label='starting point')
    plt.plot(centers[-1,0,max_id], centers[-1,1,max_id], 'ks', ms=3, label='ending point')
    plt.xlim(0,114)
    plt.ylim(0,90)
    plt.gca().set_aspect('equal',adjustable='box')
    plt.savefig('max_' + filename, dpi=300)

    plt.figure()
    plt.plot(centers[:,0,min_id], centers[:,1,min_id], lw=0.5)
    plt.plot(centers[0,0,min_id], centers[0,1,min_id], 'ko', ms=3, label='starting point')
    plt.plot(centers[-1,0,min_id], centers[-1,1,min_id], 'ks', ms=3, label='ending point')
    plt.xlim(0,114)
    plt.ylim(0,90)
    plt.gca().set_aspect('equal',adjustable='box')
    plt.savefig('min_' + filename, dpi=300)

#bins = 10
    #for iind in range(bins):
        #color = [1.0/(bins+1)*(iind+1),1.0/(bins+1)*(iind+1),1.0/(bins+1)*(iind+1)]
        #plt.plot(centers[len(centers)//bins*(iind):len(centers)//bins*(iind+1)+1,0,mol_id-1], centers[len(centers)//bins*(iind):len(centers)//bins*(iind+1)+1,1,mol_id-1], color=color, lw=0.5)

