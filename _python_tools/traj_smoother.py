import numpy as np
from csaps import csaps
"""
The opt struct can be used to set the parameters
  opt.SMOOTHPLOTS  if true, plots of all quantities will be shown
  opt.SPLINE       if true, a splien based interpolation instead of
                   Savitzky-Golay will be used. Good for rapidly varying data
                   and also performs outlier rejection
  opt.INTERPOLATE  if true (default) the returned array will be uniformly
                   sampled in time. Only available with SPLINE.
 Note that the column order of data must be
 ROS Bag
   1   time
   2    field.header.seq
   3    field.header.stamp
   4    field.header.frame_id
   5    field.pose.position.x
   6    field.pose.position.y
   7    field.pose.position.z
   8   field.pose.orientation.x
   9    field.pose.orientation.y
   10   field.pose.orientation.z
   11   field.pose.orientation.w
 The return array has the following column order:
   1   time (s)
   2   angular acceleration body x
   3   angular acceleration body y
   4   angular acceleration body z
   5   angular velocity body x
   6   angular velocity body y
   7   angular velocity body z
   8   quaterion q.x 
   9   quaterion q.y
   10  quaterion q.z
   11  quaterion q.w
   12  acceleration x
   13  acceleration y
   14  acceleration z
   15  velocity x
   16  velocity y
   17  velocity z
   18  position x
   19  position y
   20  position z
 The smoothing is done using a savitzky-golay filter of different order if the 
 corresponding parameter is not set to SPLINE.
   Position: 4th order, window 61
   Velocity: 3rd order, window 45
   Acceleration: 2nd order, window 31
   Attitude: 6th order, window 41
   Attitude Rates: 4th order, window 31
   Angular Acceleration: 2nd order, window 31
"""
def smoother(data, t, opt, dt=None):
    if opt["INTERPOLATE"] and not opt["SPLINE"]:
        print("Interpolation only available with Splines. Ignoring it.")
        opt["INTERPOLATE"] = False

    if dt is None:
        dt = 1.0 / np.round(1.0/np.mean(np.diff(data[:,2])))

    if opt["INTERPOLATE"]:
        tstart = np.min(data[:, 2])
        tend = np.max(data[:, 2])
        t = np.arange(tstart, tend, dt, dtype=np.float64)
    else:
        t = data[:, 2]

    if opt["SPLINE"]:
        # Smoothing spline smoothing coefficients
        smooth_0 = 2.5e-9 / dt
        smooth_0_reject = 1e-7 / dt
        smooth_1 = 1e-7 / dt
        smooth_2 = 1e-7 / dt

    den = []
    for i in range(0, 7):
        x = data[:, 4+i]

        if opt["SPLINE"]:
            # Fit a smoothing spline through data and calculate the residuals
            res = csaps(data[:, 2], x, data[:, 2],
                        smooth=(1.0-smooth_0_reject))-x
            idx_bad = np.abs(res) > (3.0 * np.std(res))
            # Reject outliers with residual bigger than 3 std and refit
            x = csaps(data[~idx_bad, 2], x[~idx_bad], t, smooth=(1.0-smooth_0))

            dx = np.gradient(x, t)
            dx = csaps(t, dx, t, smooth=(1.0-smooth_1))
            d2x = np.gradient(dx, t)
            d2x = csaps(t, d2x, t, smooth=(1.0-smooth_2))
        else:
            raise NotImplementedError("Method not implemented yet")

        if i == 2:
            d2x += 9.81

        den.append(np.column_stack((d2x, dx, x)))

    denoised = np.zeros((t.shape[0], 20), dtype=np.float64)
    denoised[:, 0] = t

    for i in range(0, 3):
        tmp = den[i]
        denoised[:, 11+np.array([i, 3+i, 6+i])] = tmp

    q = np.zeros([t.shape[0], 0])
    qd = np.zeros([t.shape[0], 0])
    qdd = np.zeros([t.shape[0], 0])
    for i in range(0, 4):
        tmp = den[i+3]

        q = np.column_stack([q, tmp[:, 2]])
        qd = np.column_stack([qd, tmp[:, 1]])
        qdd = np.column_stack([qdd, tmp[:, 0]])

    denoised[:, 7:11] = q

    def qinv(q):
        conjugate = q.copy()
        conjugate[:, :3] *= -1

        norm_sq = np.sum(q**2, axis=1, keepdims=True)

        return conjugate / norm_sq

    def qmult(p, q):
        x1, y1, z1, w1 = p[:, 0], p[:, 1], p[:, 2], p[:, 3]
        x2, y2, z2, w2 = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2
        w = w1*w2 - x1*x2 - y1*y2 - z1*z2

        return np.stack((x, y, z, w), axis=1)

    norms = np.linalg.norm(q, axis=1)
    inv_norms = (1.0 / norms)[:, None]
    q *= inv_norms
    qd *= inv_norms
    qdd *= inv_norms

    qi = qinv(q)
    omega = 2.0 * qmult(qi, qd)
    omega_dot = 2.0 * (qmult(qi, qdd) - 0.25 * qmult(omega, omega))

    denoised[:, 1:4] = omega_dot[:, 0:3]
    denoised[:, 4:7] = omega[:, 0:3]

    if opt["SMOOTHPLOTS"]:
        raise NotImplementedError("Feature not implemented yet")

    return denoised

def correct_quaternion(data, idx_bad=None):
    v1 = np.sum(np.abs(data[:, 7:11] - np.roll(data[:, 7:11], 1, axis=0)), axis=1)
    v2 = np.sum(np.abs(data[:, 7:11] + np.roll(data[:, 7:11], 1, axis=0)), axis=1)
    
    if idx_bad is None:
        flip_idx = (v2 < 0.95 * v1)
    else:
        flip_idx = (v2 < 0.95 * v1) * ~idx_bad
    flip_idx = np.mod(np.cumsum(flip_idx), 2)
    
    data[:, 7:11] *= (2.0 * flip_idx - 1.0)[:, None]
    return data
    
