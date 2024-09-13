
function alt_scraper {
    // Define boundaries for latitude and longitude
    local parameter start_lat.
    local parameter end_lat.

    local parameter start_long.
    local parameter end_long.

    local parameter step_size is 1/60.

    // Iterate over latitudes
    from { local lat is start_lat. } until lat >= end_lat  step { set lat to lat + step_size. } do {
        // Iterate over longitudes
        from { local long is start_long. } until long >= end_long  step { set long to long + step_size. } do {
            // Obtain terrain height
            local local_height is latlng(lat, long):terrainheight.

            // Log latitude, longitude, and terrain height to a CSV file
            log lat + "." + long + "," + local_height to terrainheight.csv.
        }
    }
}

alt_scraper(0,45, -70, -40).

function lambert_solver{
    
    // A lambert solver utilizing universal variable formulation

    local parameter r1.
    local parameter r2.
    local parameter tof.
    local parameter mu.
    local parameter t_m.
    local parameter psi is 0.
    local parameter psi_u is 4 * constant():pi^2.
    local parameter psi_l is - 4 * constant():pi.
    local parameter max_iter is 1000.
    local parameter tol is 1e-10.
    
    local function c_2{
        local parameter z.

        local function cosh {
            parameter x.
            return (constant:e^(x) + constant:e^ (-x)) / 2.
        }
        if z > 0 {
            return (1 - cos(constant:radtodeg * sqrt(z))) / z.
        }
        if z < 0 {
            return (1 - cosh(-z)) / -z.
        }
        else {
            return 1/2 .
        }
    }

    local function c_3 {
        local parameter z.
        local function sinh{
            local parameter x.
            return (constant:e^(x) - constant:e^ (-x)) / 2.
        }
        if z > 0 {
            return (sqrt(z) - sin(constant:radtodeg * sqrt(z))) / sqrt(z)^3.
        }
        if z < 0 {
            return (sinh(sqrt(-z)) - sqrt(-z)) / sqrt(-z)^3.
        }
        else {
            return 1/6 .
        }
    }

    local mag_r1 to r1:mag.
    local mag_r2 to  r2:mag.

    local gamma to vdot(r1,r2) / (mag_r1 * mag_r2).
    local A to t_m * sqrt(mag_r1 * mag_r2 * (1  + gamma)).

    if A = 0 {
        print "Orbit cannot exist".
        return list(v(0,0,0), v(0,0,0)).
    }

    local c2 to 0.5.
    local c3 to 1/6.

    local solved to false.

    from { local i is 0.} until i >= max_iter step { set i to i + 1.} do {
        set B to mag_r1 + mag_r2 + A * (psi * c3 - 1) / sqrt(c2).

        if A > 0 and B < 0 {
            set psi_l to psi_l + constant:pi.
        }

        set chi3 to sqrt( B / c2 )^3.
        set tof_ to (chi3 * c3 * A * sqrt(B)) / sqrt(mu).

        if abs(tof - tof_) < tol {
            set solved to true.
            break.
        }

        if tof_ <= tof {
            set psi_l to psi.
        } else {
            set psi_u to psi.
        }

        set psi to (psi_u + psi_l)/2.
        set c2 to c_2(psi).
        set c3 to c_3(psi).
    }

    if not solved {
        print "Did not converge".
        return list(v(0,0,0), v(0,0,0)).
    }

    local f to 1 - B / mag_r1.
    local g to A * sqrt( B / mu).
    local g_dot to 1 - B / mag_r2.
    local f_dot to (f * g_dot - 1) / g.

    local v1 to (r2 - f * r1) / g.
    local v2 to f_dot * r1 + g_dot * v1.

    return list(v1,v2).
}
