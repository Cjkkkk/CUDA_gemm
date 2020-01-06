import sys
from gpu_data import GPU_data

def calculator(Threads_Per_Block, Registers_Per_Thread, Shared_Memory_Per_Block, Compute_Capability):
    # cal active warps per SM
    Data = GPU_data[Compute_Capability]
    Registers_Per_Block = Registers_Per_Thread * Threads_Per_Block
    Block_Num_Bound_By_Reg = Data["Register File Size / SM (32-bit registers)"] // Registers_Per_Block
    Block_Num_Bound_By_Shared_Memory = Data["Register File Size / SM (32-bit registers)"] // Shared_Memory_Per_Block
    Active_Block_Num = min(Block_Num_Bound_By_Reg, Block_Num_Bound_By_Shared_Memory)
    Warps_Per_Block = (Threads_Per_Block + Data["Thread / Warp"] - 1) // Data["Thread / Warp"]
    Active_Warps = Active_Block_Num * Warps_Per_Block
    Occupancy = Active_Warps / Data["Warps / SM"]
    return Occupancy

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print ("usage: python occupancy.py [a] [b] [c] [d]")
        exit(0)

    Threads_Per_Block = int(sys.argv[1])
    Registers_Per_Thread = int(sys.argv[2])
    Shared_Memory_Per_Block = int(sys.argv[3])
    Compute_Capability = sys.argv[4]
    Occupancy = calculator(Threads_Per_Block, Registers_Per_Thread, Shared_Memory_Per_Block, Compute_Capability) * 100
    print("Occupancy = {}".format(Occupancy))