import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize
import matplotlib.cm as cm

def graphene_hamiltonian(kx, ky, t1, t2, phi, Delta, a):

    f_k = -t1 * np.exp(1j * kx * a) * (1 + 2 * np.exp(1j * 1.5 * a * kx) * 
                                        np.cos((np.sqrt(3) * a * ky) / 2))

    
    term1 = np.cos(-np.sqrt(3) * a * ky + phi)
    term2 = np.cos((3 * a / 2) * kx + (np.sqrt(3) * a / 2) * ky + phi)
    term3 = np.cos(-(3 * a / 2) * kx + (np.sqrt(3) * a / 2) * ky + phi)
    HAA = -2 * t2 * (term1 + term2 + term3) + Delta


    term1 = np.cos(np.sqrt(3) * a * ky + phi)
    term2 = np.cos(-(3 * a / 2) * kx - (np.sqrt(3) * a / 2) * ky + phi)
    term3 = np.cos((3 * a / 2) * kx - (np.sqrt(3) * a / 2) * ky + phi)
    HBB = -2 * t2 * (term1 + term2 + term3) - Delta


    
    H_k = np.array([[HAA, f_k], [np.conj(f_k), HBB]], dtype=complex)
    return H_k

def get_berry_phase(vec_k1, vec_k2, vec_k3, vec_k4, t1, t2, phi, Delta, a):
    
    u1 = np.linalg.eigh(graphene_hamiltonian(vec_k1[0], vec_k1[1], t1, t2, phi, Delta, a))[1]
    u2 = np.linalg.eigh(graphene_hamiltonian(vec_k2[0], vec_k2[1], t1, t2, phi, Delta, a))[1]
    u3 = np.linalg.eigh(graphene_hamiltonian(vec_k3[0], vec_k3[1], t1, t2, phi, Delta, a))[1]
    u4 = np.linalg.eigh(graphene_hamiltonian(vec_k4[0], vec_k4[1], t1, t2, phi, Delta, a))[1]

    dot_prod12 = np.vdot(u1[:, 1], u2[:, 1])
    dot_prod23 = np.vdot(u2[:, 1], u3[:, 1])
    dot_prod34 = np.vdot(u3[:, 1], u4[:, 1])
    dot_prod41 = np.vdot(u4[:, 1], u1[:, 1])

    berry_phase = np.angle(dot_prod12 * dot_prod23 * dot_prod34 * dot_prod41
                            / np.abs(dot_prod12 * dot_prod23 * dot_prod34 * dot_prod41))

    tmp = berry_phase / np.linalg.norm(vec_k1 - vec_k2) * np.linalg.norm(vec_k3 - vec_k4) # phi/Area of plaquette'flux density'
    
    return tmp, berry_phase

def get_berry_curvature(t1, t2, phi, Delta,a ):
    kx_array, ky_array, b_curvature  = list(), list(), list()

    k_step = 0.05

    for kx in np.arange(-np.pi, np.pi, k_step):
        for ky in np.arange(-np.pi, np.pi, k_step):
            vec_k1 = np.array([kx, ky])
            vec_k2 = np.array([kx - k_step, ky])
            vec_k3 = np.array([kx - k_step, ky - k_step])
            vec_k4 = np.array([kx, ky - k_step])

            b_curvature.append(get_berry_phase(vec_k1, vec_k2, vec_k3, vec_k4, t1, t2, phi, Delta, a)[0])

            kx_array.append(kx)
            ky_array.append(ky)

    return kx_array, ky_array, b_curvature

def calc_chern(t1, t2, phi, Delta, a):
    k_step = 0.1
    chern_number = 0
    for kx in np.arange(-np.pi, np.pi, k_step):
        for ky in np.arange(-np.pi, np.pi, k_step):
            vec_k1 = np.array([kx, ky])
            vec_k2 = np.array([kx - k_step, ky])
            vec_k3 = np.array([kx - k_step, ky - k_step])
            vec_k4 = np.array([kx, ky - k_step])

            berry_phase = get_berry_phase(vec_k1, vec_k2, vec_k3, vec_k4, t1, t2, phi, Delta, a)[1]

            normalization_factor = 1/(2*np.pi)
            chern_number += berry_phase * normalization_factor
            
    return chern_number



def berry_connection(kx, ky, t1, t2, phi, Delta, a, dk=0.1):

    H = graphene_hamiltonian(kx, ky, t1, t2, phi, Delta, a)
    eigenvalues, eigenstates = np.linalg.eigh(H)

    u = eigenstates[:, 0]

    du_dkx = (np.linalg.eigh(graphene_hamiltonian(kx + dk, ky, t1, t2, phi, Delta, a))[1][:, 0] - u) / dk
    du_dky = (np.linalg.eigh(graphene_hamiltonian(kx, ky + dk, t1, t2, phi, Delta, a))[1][:, 0] - u) / dk

    A_kx = -np.imag(np.vdot(u, du_dkx))
    A_ky = -np.imag(np.vdot(u, du_dky))

    return A_kx, A_ky

def plot_berry_connection(t1, t2, phi, Delta, a, dk=0.1, k_step=0.1):
    kx_vals, ky_vals, A_kx_vals, A_ky_vals = [], [], [], []

    for kx in np.arange(-np.pi, np.pi, k_step):
        for ky in np.arange(-np.pi, np.pi, k_step):
            A_kx, A_ky = berry_connection(kx, ky, t1, t2, phi, Delta, a, dk=dk)
            
            kx_vals.append(kx)
            ky_vals.append(ky)
            A_kx_vals.append(A_kx)
            A_ky_vals.append(A_ky)


    plt.quiver(
        kx_vals, ky_vals, A_kx_vals/np.sqrt(max(A_kx_vals)**2+max(A_ky_vals)**2), A_ky_vals/np.sqrt(max(A_kx_vals)**2+max(A_ky_vals)**2), 
        angles='xy', scale_units='xy', scale=5, 
        color='black', alpha=0.7
    )

    plt.axis("off")
    plt.show()


def plot_energy_dispersion_path(t1, t2, phi, Delta, a, num_points=100):
    Gamma = np.array([0, 0])
    K = np.array([2 * np.pi / (3 * a), 2 * np.pi / (3 * np.sqrt(3) * a)])
    K_prime = np.array([2 * np.pi / (3 * a), -2 * np.pi / (3 * np.sqrt(3) * a)])
    M = (K+K_prime)/2
    
    path = np.concatenate([
        np.linspace(M, Gamma, num_points),
        np.linspace(Gamma, K, num_points),
        np.linspace(K, M, num_points),
    ])
    
    energies = []
    for kx, ky in path:
        H_k = graphene_hamiltonian(kx, ky, t1, t2, phi, Delta, a)
        eigenvalues = np.linalg.eigvalsh(H_k)
        energies.append(eigenvalues)
    
    energies = np.array(energies)
    
    plt.plot(energies[:, 0], color='black')
    plt.plot(energies[:, 1], color='black')
    

    tick_positions = [0, num_points, 2 * num_points, 3 * num_points]
    tick_labels = ["M", "Γ", "K", "M"]

    plt.xticks(tick_positions, tick_labels, fontsize=12)

    plt.title("Energy Dispersion along path")
    plt.xlabel("Path through Brillouin Zone")
    plt.ylabel("Energy")
    plt.grid()
    plt.show()



def plot_energy_and_berry_curvature_path(t1, t2, phi, Delta, a, num_points=100):
    Gamma = np.array([0, 0])
    K = np.array([2 * np.pi / (3 * a), 2 * np.pi / (3 * np.sqrt(3) * a)])
    K_prime = np.array([2 * np.pi / (3 * a), -2 * np.pi / (3 * np.sqrt(3) * a)])
    M = (K+K_prime)/2

    path = np.concatenate([
        np.linspace(Gamma, K, num_points),
        np.linspace(K, M, num_points),
        np.linspace(M, K_prime, num_points),
        np.linspace(K_prime, Gamma, num_points),
    ])

    energies = []
    berry_curvature_values = []

    for kx, ky in path:
        H_k = graphene_hamiltonian(kx, ky, t1, t2, phi, Delta, a)
        eigenvalues = np.linalg.eigvalsh(H_k)
        energies.append(eigenvalues)

        vec_k1 = np.array([kx, ky])
        vec_k2 = np.array([kx - 0.01, ky])
        vec_k3 = np.array([kx - 0.01, ky - 0.01])
        vec_k4 = np.array([kx, ky - 0.01])
        berry_curvature, _ = get_berry_phase(vec_k1, vec_k2, vec_k3, vec_k4, t1, t2, phi, Delta, a)
        berry_curvature_values.append(berry_curvature)

    energies = np.array(energies)
    fig, ax1 = plt.subplots()

    ax1.plot(energies[:, 0], color='black')
    ax1.plot(energies[:, 1], color='black')
    ax1.set_xticks([0, num_points, 2 * num_points, 3*num_points, 4 * num_points])
    ax1.set_xticklabels(["Γ", "K", "M", "K'", "Γ"], fontsize=12)
    ax1.set_xlabel("Path through Brillouin Zone", fontsize=12)
    ax1.set_ylabel("Energy", fontsize=12)
    ax1.grid()

    ax2 = ax1.twinx()
    ax2.plot(berry_curvature_values, color='black',linestyle="--",alpha=0.3)
    ax2.set_ylabel("Berry Curvature", fontsize=12, color='grey')
    ax2.tick_params(axis='y', labelcolor='grey')

    fig.suptitle(f"Energy Dispersion alongside Berry Curvature with \n $\Delta = {Delta}$, $t_1 = {t1}$, $t_2 = {t2}$, $a = {a}$, and $\\phi = {phi:.2f}$", fontsize=14)

    plt.show()

t1 = 1
t2 = 0
phi = 0
Delta_values = [0.1, 0.4]
a = 1

# Set up the subplots
fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

for ax, Delta in zip(axes, Delta_values):
    x, y, z = get_berry_curvature(t1, t2, phi, Delta, a)
    scatter = ax.scatter(x, y, c=z, cmap='inferno', s=10)
    ax.set_title(f"Semenoff mass, $\Delta = {Delta}$", fontsize=12)
    ax.set_xlabel("$k_x$")
    ax.set_ylabel("$k_y$")
    ax.set_aspect('equal')

# Add a shared colorbar
cbar = fig.colorbar(scatter, ax=axes, location='right', shrink=0.8)
cbar.set_label("Berry Curvature", fontsize=12)

# Overall title for the figure
fig.suptitle(f"Berry Curvature in 2D for varying $\Delta$ values", fontsize=16)

plt.show()


#plot_energy_and_berry_curvature_path(t1=t1, t2=t2, phi=phi, Delta=Delta, a=a)

#plot_energy_dispersion_path(t1=1.0, t2=0, phi=0, Delta=0, a=1.0)

#plot_berry_connection(t1, t2, phi, Delta, a)


#plt.show()


#c = calc_chern(t1=t1,t2=t2,phi=phi,Delta=Delta,a=a)
#print(c)


#cherns = []
#t2s = np.linspace(-0.5,0.5,1)



#for i in t2s:
#    c = calc_chern(t1=t1,t2=i,phi=phi,Delta=Delta,a=a)
    
#    cherns.append(c)

#plt.axvline(Delta/(np.sin(phi)*3*np.sqrt(3)))
#plt.axvline(-Delta/(np.sin(phi)*3*np.sqrt(3)))

#plt.scatter(t2s,cherns)

#plt.title(f"Chern number with varying $t2$")
#plt.xlabel(f"t_2")
#plt.ylabel("Chern number")

#plt.show()


#t1=1
#t2=0
#phi=0
#Delta=0.4
#a=1


#ax = plt.axes(projection='3d')

#ax.plot_trisurf(x, y, z, cmap="viridis")

#plt.title("Berry curvature 3D")

#plt.show()
