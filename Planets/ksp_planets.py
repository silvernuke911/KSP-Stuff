
class Kerbin:
    def __init__(self):
        self.name = "Kerbin"
        
        # Orbital Characteristics
        self.semi_major_axis = 13599840256  # in meters
        self.apoapsis = 13599840256  # in meters
        self.periapsis = 13599840256  # in meters
        self.orbital_eccentricity = 0
        self.orbital_inclination = 0  # in degrees
        self.argument_of_periapsis = 0  # in degrees
        self.longitude_of_ascending_node = 0  # in degrees
        self.mean_anomaly = 3.14  # in radians
        self.sidereal_orbital_period = 9203545  # in seconds
        self.synodic_orbital_period = None  # Not defined
        self.orbital_velocity = 9285  # in m/s

        # Physical Characteristics
        self.equatorial_radius = 600000  # in meters
        self.equatorial_circumference = 3769911  # in meters
        self.surface_area = 4.5238934e12  # in square meters
        self.mass = 5.2915158e22  # in kilograms
        self.standard_gravitational_parameter = 3.5316000e12  # in m^3/s^2
        self.density = 58484.090  # in kg/m^3
        self.surface_gravity = 9.81  # in m/s^2
        self.escape_velocity = 3431.03  # in m/s
        self.sidereal_rotation_period = 21549.425  # in seconds
        self.solar_day = 21600.000  # in seconds
        self.sidereal_rotational_velocity = 174.94  # in m/s
        self.synchronous_orbit = 2863330  # in meters
        self.sphere_of_influence = 84159286  # in meters

        # Atmospheric Characteristics
        self.atmosphere_present = True
        self.atmospheric_pressure = 101.325  # in kPa
        self.atmospheric_height = 70000  # in meters
        self.temperature_min = -86.20  # in Celsius
        self.temperature_max = 15  # in Celsius
        self.oxygen_present = True

        # Scientific Multiplier
        self.scientific_multiplier_surface = 0.3
        self.scientific_multiplier_splashed = 0.4
        self.scientific_multiplier_lower_atmosphere = 0.7
        self.scientific_multiplier_upper_atmosphere = 0.9
        self.scientific_multiplier_near_space = 1
        self.scientific_multiplier_outer_space = 1.5
        self.scientific_multiplier_recovery = 1

class Mun:
    def __init__(self):
        self.name = "Mun"

        # Orbital Characteristics
        self.semi_major_axis = 12000000  # in meters
        self.apoapsis = 12000000  # in meters
        self.periapsis = 12000000  # in meters
        self.orbital_eccentricity = 0
        self.orbital_inclination = 0  # in degrees
        self.argument_of_periapsis = 0  # in degrees
        self.longitude_of_ascending_node = 0  # in degrees
        self.mean_anomaly = 1.7  # in radians
        self.sidereal_orbital_period = 138984  # in seconds
        self.synodic_orbital_period = 141115.4  # in seconds
        self.orbital_velocity = 543  # in m/s
        self.longest_time_eclipsed = 2213  # in seconds

        # Physical Characteristics
        self.equatorial_radius = 200000  # in meters
        self.equatorial_circumference = 1256637  # in meters
        self.surface_area = 5.0265482e11  # in square meters
        self.mass = 9.7599066e20  # in kilograms
        self.standard_gravitational_parameter = 6.5138398e10  # in m^3/s^2
        self.density = 29125.076  # in kg/m^3
        self.surface_gravity = 1.63  # in m/s^2
        self.escape_velocity = 807.08  # in m/s
        self.sidereal_rotation_period = 138984.38  # in seconds
        self.sidereal_rotational_velocity = 9.0416  # in m/s
        self.synchronous_orbit = 2970560  # in meters
        self.sphere_of_influence = 2429559.1  # in meters

        # Atmospheric Characteristics
        self.atmosphere_present = False

        # Scientific Multiplier
        self.scientific_multiplier_surface = 4
        self.scientific_multiplier_splashed = None  # N/A
        self.scientific_multiplier_near_space = 3
        self.scientific_multiplier_outer_space = 2
        self.scientific_multiplier_recovery = 2

class Minmus:
    def __init__(self):
        self.name = "Minmus"

        # Orbital Characteristics
        self.semi_major_axis = 47000000  # in meters
        self.apoapsis = 47000000  # in meters
        self.periapsis = 47000000  # in meters
        self.orbital_eccentricity = 0
        self.orbital_inclination = 6  # in degrees
        self.argument_of_periapsis = 38  # in degrees
        self.longitude_of_ascending_node = 78  # in degrees
        self.mean_anomaly = 0.9  # in radians
        self.sidereal_orbital_period = 1077311  # in seconds
        self.synodic_orbital_period = 1220131.7  # in seconds
        self.orbital_velocity = 274  # in m/s
        self.longest_time_eclipsed = 4378  # in seconds

        # Physical Characteristics
        self.equatorial_radius = 60000  # in meters
        self.equatorial_circumference = 376991  # in meters
        self.surface_area = 4.5238934e10  # in square meters
        self.mass = 2.6457580e19  # in kilograms
        self.standard_gravitational_parameter = 1.7658000e9  # in m^3/s^2
        self.density = 29242.046  # in kg/m^3
        self.surface_gravity = 0.491  # in m/s^2
        self.escape_velocity = 242.61  # in m/s
        self.sidereal_rotation_period = 40400.000  # in seconds
        self.sidereal_rotational_velocity = 9.3315  # in m/s
        self.synchronous_orbit = 357940  # in meters
        self.sphere_of_influence = 2247428.4  # in meters

        # Atmospheric Characteristics
        self.atmosphere_present = False

        # Scientific Multiplier
        self.scientific_multiplier_surface = 5
        self.scientific_multiplier_splashed = None  # N/A
        self.scientific_multiplier_near_space = 4
        self.scientific_multiplier_outer_space = 2.5
        self.scientific_multiplier_recovery = 2.5

class Kerbol:
    def __init__(self):
        self.name = "Kerbol"

        # Physical Characteristics
        self.equatorial_radius = 261600000  # in meters
        self.equatorial_circumference = 1643681276  # in meters
        self.surface_area = 8.5997404e17  # in square meters
        self.mass = 1.7565459e28  # in kilograms
        self.standard_gravitational_parameter = 1.1723328e18  # in m^3/s^2
        self.density = 234.23818  # in kg/m^3
        self.surface_gravity = 17.1  # in m/s^2
        self.escape_velocity = 94672.01  # in m/s
        self.sidereal_rotation_period = 432000.00  # in seconds
        self.sidereal_rotational_velocity = 3804.8  # in m/s
        self.synchronous_orbit = 1508045290  # in meters
        self.sphere_of_influence = float('inf')  # Sphere of influence is infinite

        # Atmospheric Characteristics
        self.atmosphere_present = True
        self.atmospheric_pressure = 16.0  # in kPa
        self.atmospheric_pressure_atm = 0.157908  # in atm
        self.atmospheric_height = 600000  # in meters
        self.temperature_min = 3741.80  # in °C
        self.temperature_min_k = 4014.95  # in K
        self.temperature_max = 9348.05  # in °C
        self.temperature_max_k = 9621.2  # in K
        self.oxygen_present = False

        # Scientific Multiplier
        self.scientific_multiplier_surface = None  # N/A
        self.scientific_multiplier_splashed = None  # N/A
        self.scientific_multiplier_lower_atmosphere = 1
        self.scientific_multiplier_upper_atmosphere = 1
        self.scientific_multiplier_near_space = 11
        self.scientific_multiplier_outer_space = 2
        self.scientific_multiplier_recovery = 4

class Moho:
    def __init__(self):
        self.name = "Moho"

        # Orbital Characteristics
        self.semi_major_axis = 5263138304  # in meters
        self.apoapsis = 6315765981  # in meters
        self.periapsis = 4210510628  # in meters
        self.orbital_eccentricity = 0.2
        self.orbital_inclination = 7  # in degrees
        self.argument_of_periapsis = 15  # in degrees
        self.longitude_of_ascending_node = 70  # in degrees
        self.mean_anomaly = 3.14  # in radians
        self.sidereal_orbital_period = 2215754  # in seconds
        self.synodic_orbital_period = 2918346.4  # in seconds
        self.orbital_velocity = (12186, 18279)  # in m/s (min, max)

        # Physical Characteristics
        self.equatorial_radius = 250000  # in meters
        self.equatorial_circumference = 1570796  # in meters
        self.surface_area = 7.8539816e11  # in square meters
        self.mass = 2.5263314e21  # in kilograms
        self.standard_gravitational_parameter = 1.6860938e11  # in m^3/s^2
        self.density = 38599.5  # in kg/m^3
        self.surface_gravity = 2.7  # in m/s^2
        self.escape_velocity = 1161.41  # in m/s
        self.sidereal_rotation_period = 1210000.0  # in seconds
        self.solar_day = 2665723.4  # in seconds
        self.sidereal_rotational_velocity = 1.2982  # in m/s
        self.synchronous_orbit = 18173.17  # in meters
        self.sphere_of_influence = 9646663.0  # in meters

        # Atmospheric Characteristics
        self.atmosphere_present = False

        # Scientific Multiplier
        self.scientific_multiplier_surface = 10
        self.scientific_multiplier_splashed = None
        self.scientific_multiplier_near_space = 8
        self.scientific_multiplier_outer_space = 7
        self.scientific_multiplier_recovery = 7

class Eve:
    def __init__(self):
        self.name = "Eve"

        # Orbital Characteristics
        self.semi_major_axis = 9832684544  # in meters
        self.apoapsis = 9931011387  # in meters
        self.periapsis = 9734357701  # in meters
        self.orbital_eccentricity = 0.01
        self.orbital_inclination = 2.1  # in degrees
        self.argument_of_periapsis = 0  # in degrees
        self.longitude_of_ascending_node = 15  # in degrees
        self.mean_anomaly = 3.14  # in radians
        self.sidereal_orbital_period = 5657995  # in seconds
        self.synodic_orbital_period = 14687035.5  # in seconds
        self.orbital_velocity = (10811, 11029)  # in m/s (min, max)

        # Physical Characteristics
        self.equatorial_radius = 700000  # in meters
        self.equatorial_circumference = 4398230  # in meters
        self.surface_area = 6.1575216e12  # in square meters
        self.mass = 1.2243980e23  # in kilograms
        self.standard_gravitational_parameter = 8.1717302e12  # in m^3/s^2
        self.density = 85219.677  # in kg/m^3
        self.surface_gravity = 16.7  # in m/s^2
        self.escape_velocity = 4831.96  # in m/s
        self.sidereal_rotation_period = 80500.000  # in seconds
        self.solar_day = 81661.857  # in seconds
        self.sidereal_rotational_velocity = 54.636  # in m/s
        self.synchronous_orbit = 10328.47  # in meters
        self.sphere_of_influence = 85109365  # in meters

        # Atmospheric Characteristics
        self.atmosphere_present = True
        self.atmospheric_pressure = 506.625  # in kPa
        self.atmospheric_height = 90000  # in meters
        self.temperature_min = -113.13  # in Celsius
        self.temperature_max = 146.85  # in Celsius
        self.oxygen_present = False

        # Scientific Multiplier
        self.scientific_multiplier_surface = 8
        self.scientific_multiplier_splashed = 8
        self.scientific_multiplier_lower_atmosphere = 6
        self.scientific_multiplier_upper_atmosphere = 6
        self.scientific_multiplier_near_space = 7
        self.scientific_multiplier_outer_space = 5
        self.scientific_multiplier_recovery = 5

class Gilly:
    def __init__(self):
        self.name = "Gilly"

        # Orbital Characteristics
        self.semi_major_axis = 31500000  # in meters
        self.apoapsis = 48825000  # in meters
        self.periapsis = 14175000  # in meters
        self.orbital_eccentricity = 0.55
        self.orbital_inclination = 12  # in degrees
        self.argument_of_periapsis = 10  # in degrees
        self.longitude_of_ascending_node = 80  # in degrees
        self.mean_anomaly = 0.9  # in radians
        self.sidereal_orbital_period = 388587  # in seconds
        self.synodic_orbital_period = 417243.4  # in seconds
        self.orbital_velocity = (274, 945)  # in m/s (min, max)
        self.longest_time_eclipsed = 5102  # in seconds

        # Physical Characteristics
        self.equatorial_radius = 13000  # in meters
        self.equatorial_circumference = 81681  # in meters
        self.surface_area = 2.1237166e9  # in square meters
        self.mass = 1.2420363e17  # in kilograms
        self.standard_gravitational_parameter = 8289449.8  # in m^3/s^2
        self.density = 13496.328  # in kg/m^3
        self.surface_gravity = 0.049  # in m/s^2
        self.escape_velocity = 35.71  # in m/s
        self.sidereal_rotation_period = 28255.000  # in seconds
        self.sidereal_rotational_velocity = 2.8909  # in m/s
        self.synchronous_orbit = 42140  # in meters
        self.sphere_of_influence = 126123  # in meters

        # Atmospheric Characteristics
        self.atmosphere_present = False

        # Scientific Multiplier
        self.scientific_multiplier_surface = 5
        self.scientific_multiplier_splashed = None
        self.scientific_multiplier_near_space = 5
        self.scientific_multiplier_outer_space = 5
        self.scientific_multiplier_recovery = 5

class Duna:
    def __init__(self):
        self.name = "Duna"

        # Orbital Characteristics
        self.semi_major_axis = 20726155264  # in meters
        self.apoapsis = 21783189163  # in meters
        self.periapsis = 19669121365  # in meters
        self.orbital_eccentricity = 0.051
        self.orbital_inclination = 0.06  # in degrees
        self.argument_of_periapsis = 0  # in degrees
        self.longitude_of_ascending_node = 135.5  # in degrees
        self.mean_anomaly = 3.14  # in radians
        self.sidereal_orbital_period = 17315400  # in seconds
        self.synodic_orbital_period = 19645697.3  # in seconds
        self.orbital_velocity = (7147, 7915)  # in m/s (min, max)

        # Physical Characteristics
        self.equatorial_radius = 320000  # in meters
        self.equatorial_circumference = 2010619  # in meters
        self.surface_area = 1.2867964e12  # in square meters
        self.mass = 4.515427e21  # in kilograms
        self.standard_gravitational_parameter = 3.0136321e11  # in m^3/s^2
        self.density = 32897.302  # in kg/m^3
        self.surface_gravity = 2.94  # in m/s^2
        self.escape_velocity = 1372.41  # in m/s
        self.sidereal_rotation_period = 65517.859  # in seconds
        self.solar_day = 65766.707  # in seconds
        self.sidereal_rotational_velocity = 30.688  # in m/s
        self.synchronous_orbit = 2880000  # in meters
        self.sphere_of_influence = 47921949  # in meters

        # Atmospheric Characteristics
        self.atmosphere_present = True
        self.atmospheric_pressure = 6.755  # in kPa
        self.atmospheric_height = 50000  # in meters
        self.temperature_min = -123.15  # in Celsius
        self.temperature_max = -40.15  # in Celsius
        self.oxygen_present = False

        # Scientific Multiplier
        self.scientific_multiplier_surface = 8
        self.scientific_multiplier_splashed = None
        self.scientific_multiplier_lower_atmosphere = 5
        self.scientific_multiplier_upper_atmosphere = 5
        self.scientific_multiplier_near_space = 7
        self.scientific_multiplier_outer_space = 5
        self.scientific_multiplier_recovery = 5

class Ike:
    def __init__(self):
        self.name = "Ike"

        # Orbital Characteristics
        self.semi_major_axis = 3200000  # in meters
        self.apoapsis = 3296000  # in meters
        self.periapsis = 3104000  # in meters
        self.orbital_eccentricity = 0.03
        self.orbital_inclination = 0.2  # in degrees
        self.argument_of_periapsis = 0  # in degrees
        self.longitude_of_ascending_node = 0  # in degrees
        self.mean_anomaly = 1.7  # in radians
        self.sidereal_orbital_period = 65518  # in seconds
        self.synodic_orbital_period = 65766.7  # in seconds
        self.orbital_velocity = (298, 316)  # in m/s (min, max)

        # Physical Characteristics
        self.equatorial_radius = 130000  # in meters
        self.equatorial_circumference = 816814  # in meters
        self.surface_area = 2.1237166e11  # in square meters
        self.mass = 2.7821615e20  # in kilograms
        self.standard_gravitational_parameter = 1.8568369e10  # in m^3/s^2
        self.density = 30231.777  # in kg/m^3
        self.surface_gravity = 1.10  # in m/s^2
        self.escape_velocity = 534.48  # in m/s
        self.sidereal_rotation_period = 65517.862  # in seconds
        self.sidereal_rotational_velocity = 12.467  # in m/s
        self.synchronous_orbit = 1133.90  # in kilometers
        self.sphere_of_influence = 1049598.9  # in meters

        # Atmospheric Characteristics
        self.atmosphere_present = False

        # Scientific Multiplier
        self.scientific_multiplier_surface = 8
        self.scientific_multiplier_splashed = None
        self.scientific_multiplier_near_space = 7
        self.scientific_multiplier_outer_space = 5
        self.scientific_multiplier_recovery = 5

class Dres:
    def __init__(self):
        self.name = "Dres"

        # Orbital Characteristics
        self.semi_major_axis = 40839348203  # in meters
        self.apoapsis = 46761053692  # in meters
        self.periapsis = 34917642714  # in meters
        self.orbital_eccentricity = 0.145
        self.orbital_inclination = 5  # in degrees
        self.argument_of_periapsis = 90  # in degrees
        self.longitude_of_ascending_node = 280  # in degrees
        self.mean_anomaly = 3.14  # in radians
        self.sidereal_orbital_period = 47893063  # in seconds
        self.synodic_orbital_period = 11392903.3  # in seconds
        self.orbital_velocity = (4630, 6200)  # in m/s (min, max)

        # Physical Characteristics
        self.equatorial_radius = 138000  # in meters
        self.equatorial_circumference = 867080  # in meters
        self.surface_area = 2.3931396e11  # in square meters
        self.mass = 3.2190937e20  # in kilograms
        self.standard_gravitational_parameter = 2.1484489e10  # in m^3/s^2
        self.density = 29242.045  # in kg/m^3
        self.surface_gravity = 1.13  # in m/s^2
        self.escape_velocity = 558.00  # in m/s
        self.sidereal_rotation_period = 34800  # in seconds
        self.solar_day = 34825.305  # in seconds
        self.sidereal_rotational_velocity = 24.916  # in m/s
        self.synchronous_orbit = 732240  # in meters
        self.sphere_of_influence = 32832840  # in meters

        # Atmospheric Characteristics
        self.atmosphere_present = False

        # Scientific Multiplier
        self.scientific_multiplier_surface = 8
        self.scientific_multiplier_splashed = None
        self.scientific_multiplier_near_space = 7
        self.scientific_multiplier_outer_space = 6
        self.scientific_multiplier_recovery = 6

class Jool:
    def __init__(self):
        self.name = "Jool"

        # Orbital Characteristics
        self.semi_major_axis = 68773560320  # in meters
        self.apoapsis = 72212238387  # in meters
        self.periapsis = 65334882253  # in meters
        self.orbital_eccentricity = 0.05
        self.orbital_inclination = 1.304  # in degrees
        self.argument_of_periapsis = 0  # in degrees
        self.longitude_of_ascending_node = 52  # in degrees
        self.mean_anomaly = 0.1  # in radians
        self.sidereal_orbital_period = 104661432  # in seconds
        self.synodic_orbital_period = 10090901.7  # in seconds
        self.orbital_velocity = (3927, 4341)  # in m/s (min, max)

        # Physical Characteristics
        self.equatorial_radius = 6000000  # in meters
        self.equatorial_circumference = 37699112  # in meters
        self.surface_area = 4.5238934e14  # in square meters
        self.mass = 4.2332127e24  # in kilograms
        self.standard_gravitational_parameter = 2.8252800e14  # in m^3/s^2
        self.density = 4678.7273  # in kg/m^3
        self.surface_gravity = 7.85  # in m/s^2
        self.escape_velocity = 9704.43  # in m/s
        self.sidereal_rotation_period = 36000  # in seconds
        self.solar_day = 36012.387  # in seconds
        self.sidereal_rotational_velocity = 1047.2  # in m/s
        self.synchronous_orbit = 15010.46  # in kilometers
        self.sphere_of_influence = 2455985200  # in meters

        # Atmospheric Characteristics
        self.atmosphere_present = True
        self.atmospheric_pressure = 1519.88  # in kPa
        self.atmospheric_height = 200000  # in meters
        self.temperature_min = -153.14  # in Celsius
        self.temperature_max = -48.08  # in Celsius
        self.oxygen_present = False

        # Scientific Multiplier
        self.scientific_multiplier_lower_atmosphere = 12
        self.scientific_multiplier_upper_atmosphere = 9
        self.scientific_multiplier_near_space = 7
        self.scientific_multiplier_outer_space = 6
        self.scientific_multiplier_recovery = 6

class Laythe:
    def __init__(self):
        self.name = "Laythe"

        # Orbital Characteristics
        self.semi_major_axis = 27184000  # in meters
        self.apoapsis = 27184000  # in meters
        self.periapsis = 27184000  # in meters
        self.orbital_eccentricity = 0
        self.orbital_inclination = 0  # in degrees
        self.argument_of_periapsis = 0  # in degrees
        self.longitude_of_ascending_node = 0  # in degrees
        self.mean_anomaly = 3.14  # in radians
        self.sidereal_orbital_period = 52981  # in seconds
        self.synodic_orbital_period = 53007.7  # in seconds
        self.orbital_velocity = 3224  # in m/s

        # Physical Characteristics
        self.equatorial_radius = 500000  # in meters
        self.equatorial_circumference = 3141593  # in meters
        self.surface_area = 3.1415927e12  # in square meters
        self.mass = 2.9397311e22  # in kilograms
        self.standard_gravitational_parameter = 1.9620000e12  # in m^3/s^2
        self.density = 56144.728  # in kg/m^3
        self.surface_gravity = 7.85  # in m/s^2
        self.escape_velocity = 2801.43  # in m/s
        self.sidereal_rotation_period = 52980.879  # in seconds
        self.sidereal_rotational_velocity = 59.297  # in m/s
        self.synchronous_orbit = 4686.32  # in kilometers
        self.sphere_of_influence = 3723645.8  # in meters

        # Atmospheric Characteristics
        self.atmosphere_present = True
        self.atmospheric_pressure = 60.7950  # in kPa
        self.atmospheric_height = 50000  # in meters
        self.temperature_min = -74.15  # in Celsius
        self.temperature_max = 3.85  # in Celsius
        self.oxygen_present = True

        # Scientific Multiplier
        self.scientific_multiplier_surface = 14
        self.scientific_multiplier_splashed = 12
        self.scientific_multiplier_lower_atmosphere = 11
        self.scientific_multiplier_upper_atmosphere = 10
        self.scientific_multiplier_near_space = 9
        self.scientific_multiplier_outer_space = 8
        self.scientific_multiplier_recovery = 8

class Vall:
    def __init__(self):
        self.name = "Vall"

        # Orbital Characteristics
        self.semi_major_axis = 43152000  # in meters
        self.apoapsis = 43152000  # in meters
        self.periapsis = 43152000  # in meters
        self.orbital_eccentricity = 0
        self.orbital_inclination = 0  # in degrees
        self.argument_of_periapsis = 0  # in degrees
        self.longitude_of_ascending_node = 0  # in degrees
        self.mean_anomaly = 0.9  # in radians
        self.sidereal_orbital_period = 105962  # in seconds
        self.synodic_orbital_period = 106069.5  # in seconds
        self.orbital_velocity = 2559  # in m/s

        # Physical Characteristics
        self.equatorial_radius = 300000  # in meters
        self.equatorial_circumference = 1884956  # in meters
        self.surface_area = 1.1309734e12  # in square meters
        self.mass = 3.1087655e21  # in kilograms
        self.standard_gravitational_parameter = 2.0748150e11  # in m^3/s^2
        self.density = 27487.522  # in kg/m^3
        self.surface_gravity = 2.31  # in m/s^2
        self.escape_velocity = 1176.10  # in m/s
        self.sidereal_rotation_period = 105962.09  # in seconds
        self.sidereal_rotational_velocity = 17.789  # in m/s
        self.synchronous_orbit = 3593.20  # in kilometers
        self.sphere_of_influence = 2406401.4  # in meters

        # Atmospheric Characteristics
        self.atmosphere_present = False

        # Scientific Multiplier
        self.scientific_multiplier_surface = 12
        self.scientific_multiplier_splashed = None
        self.scientific_multiplier_near_space = 9
        self.scientific_multiplier_outer_space = 8
        self.scientific_multiplier_recovery = 8

class Tylo:
    def __init__(self):
        self.name = "Tylo"

        # Orbital Characteristics
        self.semi_major_axis = 68500000  # in meters
        self.apoapsis = 68500000  # in meters
        self.periapsis = 68500000  # in meters
        self.orbital_eccentricity = 0
        self.orbital_inclination = 0.025  # in degrees
        self.argument_of_periapsis = 0  # in degrees
        self.longitude_of_ascending_node = 0  # in degrees
        self.mean_anomaly = 3.14  # in radians
        self.sidereal_orbital_period = 211926  # in seconds
        self.synodic_orbital_period = 212356.4  # in seconds
        self.orbital_velocity = 2031  # in m/s

        # Physical Characteristics
        self.equatorial_radius = 600000  # in meters
        self.equatorial_circumference = 3769911  # in meters
        self.surface_area = 4.5238934e12  # in square meters
        self.mass = 4.2332127e22  # in kilograms
        self.standard_gravitational_parameter = 2.8252800e12  # in m^3/s^2
        self.density = 46787.273  # in kg/m^3
        self.surface_gravity = 7.85  # in m/s^2
        self.escape_velocity = 3068.81  # in m/s
        self.sidereal_rotation_period = 211926.36  # in seconds
        self.sidereal_rotational_velocity = 17.789  # in m/s
        self.synchronous_orbit = 14157.88  # in kilometers
        self.sphere_of_influence = 10856518  # in meters

        # Atmospheric Characteristics
        self.atmosphere_present = False

        # Scientific Multiplier
        self.scientific_multiplier_surface = 12
        self.scientific_multiplier_splashed = None  # Not applicable
        self.scientific_multiplier_near_space = 10
        self.scientific_multiplier_outer_space = 8
        self.scientific_multiplier_recovery = 8

class Bop:
    def __init__(self):
        self.name = "Bop"

        # Orbital Characteristics
        self.semi_major_axis = 128500000  # in meters
        self.apoapsis = 158697500  # in meters
        self.periapsis = 98302500  # in meters
        self.orbital_eccentricity = 0.235
        self.orbital_inclination = 15  # in degrees
        self.argument_of_periapsis = 25  # in degrees
        self.longitude_of_ascending_node = 10  # in degrees
        self.mean_anomaly = 0.9  # in radians
        self.sidereal_orbital_period = 544507  # in seconds
        self.synodic_orbital_period = 547355.1  # in seconds
        self.orbital_velocity = (1167, 1884)  # in m/s

        # Physical Characteristics
        self.equatorial_radius = 65000  # in meters
        self.equatorial_circumference = 408407  # in meters
        self.surface_area = 5.3092916e10  # in square meters
        self.mass = 3.7261090e19  # in kilograms
        self.standard_gravitational_parameter = 2.4868349e9  # in m^3/s^2
        self.density = 32391.188  # in kg/m^3
        self.surface_gravity = 0.589  # in m/s^2
        self.escape_velocity = 276.62  # in m/s
        self.sidereal_rotation_period = 544507.43  # in seconds
        self.sidereal_rotational_velocity = 0.75005  # in m/s
        self.synchronous_orbit = 2588.17  # in kilometers
        self.sphere_of_influence = 1221060.9  # in meters

        # Atmospheric Characteristics
        self.atmosphere_present = False

        # Scientific Multiplier
        self.scientific_multiplier_surface = 12
        self.scientific_multiplier_splashed = None  # Not applicable
        self.scientific_multiplier_near_space = 9
        self.scientific_multiplier_outer_space = 8
        self.scientific_multiplier_recovery = 8

class Pol:
    def __init__(self):
        self.name = "Pol"

        # Orbital Characteristics
        self.semi_major_axis = 179890000  # in meters
        self.apoapsis = 210624207  # in meters
        self.periapsis = 149155794  # in meters
        self.orbital_eccentricity = 0.171
        self.orbital_inclination = 4.25  # in degrees
        self.argument_of_periapsis = 15  # in degrees
        self.longitude_of_ascending_node = 2  # in degrees
        self.mean_anomaly = 0.9  # in radians
        self.sidereal_orbital_period = 901903  # in seconds
        self.synodic_orbital_period = 909742.2  # in seconds
        self.orbital_velocity = (1055, 1489)  # in m/s

        # Physical Characteristics
        self.equatorial_radius = 44000  # in meters
        self.equatorial_circumference = 276460  # in meters
        self.surface_area = 2.4328494e10  # in square meters
        self.mass = 1.0813507e19  # in kilograms
        self.standard_gravitational_parameter = 7.2170208e8  # in m^3/s^2
        self.density = 30305.392  # in kg/m^3
        self.surface_gravity = 0.373  # in m/s^2
        self.escape_velocity = 181.12  # in m/s
        self.sidereal_rotation_period = 901902.62  # in seconds
        self.sidereal_rotational_velocity = 0.30653  # in m/s
        self.synchronous_orbit = 2415.08  # in kilometers
        self.sphere_of_influence = 1042138.9  # in meters

        # Atmospheric Characteristics
        self.atmosphere_present = False

        # Scientific Multiplier
        self.scientific_multiplier_surface = 12
        self.scientific_multiplier_splashed = None  # Not applicable
        self.scientific_multiplier_near_space = 9
        self.scientific_multiplier_outer_space = 8
        self.scientific_multiplier_recovery = 8

class Eeloo:
    def __init__(self):
        self.name = "Eeloo"

        # Orbital Characteristics
        self.semi_major_axis = 90118820000  # in meters
        self.apoapsis = 113549713200  # in meters
        self.periapsis = 66687926800  # in meters
        self.orbital_eccentricity = 0.26
        self.orbital_inclination = 6.15  # in degrees
        self.argument_of_periapsis = 260  # in degrees
        self.longitude_of_ascending_node = 50  # in degrees
        self.mean_anomaly = 3.14  # in radians
        self.sidereal_orbital_period = 156992048  # in seconds
        self.synodic_orbital_period = 9776696.3  # in seconds
        self.orbital_velocity = (2764, 4706)  # in m/s

        # Physical Characteristics
        self.equatorial_radius = 210000  # in meters
        self.equatorial_circumference = 1319469  # in meters
        self.surface_area = 5.5417694e11  # in square meters
        self.mass = 1.1149224e21  # in kilograms
        self.standard_gravitational_parameter = 7.4410815e10  # in m^3/s^2
        self.density = 28740.754  # in kg/m^3
        self.surface_gravity = 1.69  # in m/s^2
        self.escape_velocity = 841.83  # in m/s
        self.sidereal_rotation_period = 19460.000  # in seconds
        self.solar_day = 19462.412  # in seconds
        self.sidereal_rotational_velocity = 67.804  # in m/s
        self.synchronous_orbit = 683.69  # in kilometers
        self.sphere_of_influence = 119082940  # in meters

        # Atmospheric Characteristics
        self.atmosphere_present = False

        # Scientific Multiplier
        self.scientific_multiplier_surface = 15
        self.scientific_multiplier_splashed = None  # Not applicable
        self.scientific_multiplier_near_space = 12
        self.scientific_multiplier_outer_space = 10
        self.scientific_multiplier_recovery = 10

class constants:
    def __init__(self):
        # Newton’s Gravitational Constant (m^3 kg^-1 s^-2)
        self.G = 6.67430e-11
        # Gravity acceleration (m/s^2) at sea level on Earth
        self.g0 = 9.80665
        # Speed of light in a vacuum, in m/s
        self.c = 299792458
        # Conversion constant: Atmospheres to kiloPascals
        self.AtmToKPa = 101.325
        # Conversion constant: kiloPascals to Atmospheres
        self.KPaToAtm = 1 / 101.325
        # Conversion constant: Degrees to Radians
        self.DegToRad = 0.01745329252
        # Conversion constant: Radians to Degrees
        self.RadToDeg = 57.2957795131
        # Avogadro’s Constant (mol^-1)
        self.Avogadro = 6.02214076e23
        # Boltzmann’s Constant (m^2 kg s^-2 K^-1)
        self.Boltzmann = 1.380649e-23
        # The Ideal Gas Constant (J mol^-1 K^-1)
        self.IdealGas = 8.314462618

constant=constants()
kerbol = Kerbol()
moho = Moho()
eve = Eve()
kerbin = Kerbin()
mun = Mun()
minmus = Minmus()
duna = Duna()
ike = Ike()
dres = Dres()
jool = Jool()
laythe = Laythe()
vall = Vall()
tylo = Tylo()
bop = Bop()
pol = Pol()
eeloo = Eeloo()
