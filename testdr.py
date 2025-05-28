from dr import DeadReckoning # Assuming you saved the class in this file
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt



# Define initial state 
initial_pos = np.array([0.0, 0.0, 0.0])      # Initial pos
initial_ori_quat = np.array([0.0, 0.0, 0.0, 1.0]) # Initial orientation [qx, qy, qz, qw] (identity)
#initial_vel = np.array([-1.5754074010726713, -7.1777300308009870, 1.0557594491640927])      # hard Initial velocity
initial_vel = np.array([-1.0023593573621994, -4.5579541737370999, 0.7384016557587667])      # FOR EASY


my_initial_state = {
    "ts": 0.0, # Should be optional
    "pos": initial_pos,
    "ori": Rotation.from_quat(initial_ori_quat), # scipy
    "vel": initial_vel
}

estimator = DeadReckoning(initial_state=my_initial_state)

# 3. Load IMU data
imu_file_path = "/data/storage/datasets/TartanEvent/ocean/Easy/P000/imu_left.txt"
#imu_file_path = "/data/storage/datasets/TartanEvent/ocean/Hard/P000/imu_left.txt"
estimator.load_imu_data(imu_file_path)

# 4. Estimate the pose if data was loaded successfully
if estimator.imu_data:
    trajectory_dict = estimator.estimate_pose()
    # trajectory_dict_interp = estimator_interp.estimate_pose() # For interpolation test

    if trajectory_dict:
        # Access trajectory data
        print("\nTrajectory Timestamps:\n", trajectory_dict["ts"])
        print("\nTrajectory Positions (x, y, z):\n", trajectory_dict["pos"])
        # ... (print other parts if needed)

        # Get as a single NumPy array
        trajectory_array = estimator.get_trajectory_as_numpy()
        if trajectory_array is not None:
            print("\nTrajectory as NumPy array (t, x, y, z, qx, qy, qz, qw, vx, vy, vz):\n", trajectory_array[0:5]) # Print first 5 rows
            output_traj_file = "estimated_trajectory.txt"
            np.savetxt(output_traj_file, trajectory_array, header="t x y z qx qy qz qw vx vy vz", fmt="%1.6f")
            print(f"Estimated trajectory saved to {output_traj_file}")

        # --- Plotting the trajectory ---
        positions = trajectory_dict["pos"]
        if positions.shape[0] > 1: # Need at least two points to plot a line
            plt.figure(figsize=(8, 6))
            plt.plot(positions[:, 0], positions[:, 1], marker='o', linestyle='-', label='Estimated Trajectory (X-Y)')
            plt.scatter(positions[0, 0], positions[0, 1], color='green', s=100, zorder=5, label='Start')
            plt.scatter(positions[-1, 0], positions[-1, 1], color='red', s=100, zorder=5, label='End')
            plt.xlabel("X position (m)")
            plt.ylabel("Y position (m)")
            plt.title("Estimated 2D Trajectory (X-Y Plane)")
            plt.legend()
            plt.grid(True)
            plt.axis('equal') # Equal scaling for X and Y axes
            
            plot_filename = "estimated_trajectory_plot.png"
            plt.savefig(plot_filename)
            print(f"Trajectory plot saved to {plot_filename}")
            # plt.show() # Would display the plot if not on a headless server
        else:
            print("Not enough data points in the trajectory to plot.")
            
else:
    print("Could not run pose estimation as IMU data failed to load.")