from flask import Flask, request, jsonify
import pickle
import numpy as np

# Load the model
model = pickle.load(open("model1.pkl", 'rb'))

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from the request form
    absolute_magnitude = request.form.get('absolute_magnitude')
    minimum_orbit_intersection = request.form.get('minimum_orbit_intersection')
    perihelion_time = request.form.get('perihelion_time')
    asc_node_longitude = request.form.get('asc_node_longitude')
    perihelion_distance = request.form.get('perihelion_distance')
    perihelion_arg = request.form.get('perihelion_arg')
    orbit_uncertainty = request.form.get('orbit_uncertainty')
    eccentricity = float(request.form.get('eccentricity'))
    relative_velocity_km_per_sec = request.form.get('relative_velocity_km_per_sec')
    mean_anomaly = request.form.get('mean_anomaly')
    miss_dist_astronomical = request.form.get('miss_dist_astronomical')
    jupiter_tisserand_invariant = request.form.get('jupiter_tisserand_invariant')
    aphelion_dist = request.form.get('aphelion_dist')
    inclination = request.form.get('inclination')
    mean_motion = request.form.get('mean_motion')
    semi_major_axis = request.form.get('semi_major_axis')
    orbital_period = request.form.get('orbital_period')
    epoch_osculation = request.form.get('epoch osculation')
    est_dia_in_km = request.form.get('est_dia_in_km')
    
    # Create input query array
    input_query = np.array([[absolute_magnitude, minimum_orbit_intersection, perihelion_time, asc_node_longitude, 
                            perihelion_distance, perihelion_arg, orbit_uncertainty, eccentricity, 
                            relative_velocity_km_per_sec, mean_anomaly, miss_dist_astronomical, 
                            jupiter_tisserand_invariant, aphelion_dist, inclination, mean_motion, semi_major_axis,orbital_period,epoch_osculation,est_dia_in_km]])
    
    # Make prediction
    result = model.predict(input_query)[0]
    
    # Return result as JSON
    return jsonify({'Hazardous': str(result)})

if __name__ == '__main__':
    app.run(debug=True)
