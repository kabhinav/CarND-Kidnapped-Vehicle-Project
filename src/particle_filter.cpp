/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {

  default_random_engine gen;
  num_particles = 100;

  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);

  // intialize particles
  for(int i=0; i < num_particles; i++ ){
    Particle particle;
    particle.id = i;
    particle.x = dist_x(gen);
    particle.y = dist_y(gen);
    particle.theta = dist_theta(gen);
    particle.weight = 1;
    weights.push_back(1);
    particles.push_back(particle);
  }

  is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {

  default_random_engine gen;

  // determine new particle positions
  for(int i=0; i< num_particles; i++){
    double x_t;
    double y_t;
    double theta_t;

    Particle &curr = particles[i];
    if (fabs(yaw_rate) > 1e-6){
      double yaw_dt =  yaw_rate*delta_t;
      x_t = curr.x + (velocity/yaw_rate) * (sin(curr.theta + yaw_dt) - sin(curr.theta));
      y_t = curr.y + (velocity/yaw_rate) * (cos(curr.theta) - cos(curr.theta + yaw_dt));
      theta_t = curr.theta + (yaw_rate*delta_t);
    } else {
      x_t = curr.x + velocity * delta_t * cos(curr.theta);
      y_t = curr.y + velocity * delta_t * sin(curr.theta);
      theta_t = curr.theta;
    }
    // normalize
    normal_distribution<double> dist_x(x_t, std_pos[0]);
    normal_distribution<double> dist_y(y_t, std_pos[1]);
    normal_distribution<double> dist_theta(theta_t, std_pos[2]);

    curr.x = dist_x(gen);
    curr.y = dist_y(gen);
    curr.theta = dist_theta(gen);
  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {

  for (LandmarkObs &o : observations) {
    double min_dist = numeric_limits<double>::max();
    double cur_dist = -1.0;

    for (LandmarkObs &p : predicted) {
      cur_dist = dist(p.x, p.y, o.x, o.y);
      if (min_dist > cur_dist) {
	min_dist = cur_dist;
	o.id = p.id;
      }
    }
  }
}


void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
		std::vector<LandmarkObs> observations, Map map_landmarks) {

  for(Particle &particle: particles){
    // locate nearby landmarks
    vector<LandmarkObs> obsrvd_landmarks;
    for(Map::single_landmark_s &obs: map_landmarks.landmark_list){
      if (dist(obs.x_f, obs.y_f, particle.x, particle.y) < sensor_range){
	obsrvd_landmarks.push_back({int(obsrvd_landmarks.size()), double(obs.x_f), double(obs.y_f)});
      }
    }

    vector<LandmarkObs> mapped_observations;
    double sin_theta = sin(particle.theta);
    double cos_theta = cos(particle.theta);
    for(const LandmarkObs &obs: observations){
      mapped_observations.push_back({obs.id,
	    obs.x*cos_theta - obs.y*sin_theta + particle.x,
	    obs.x*sin_theta + obs.y*cos_theta + particle.y
	});
    }

    dataAssociation(obsrvd_landmarks, mapped_observations);

    // calculate distribution weight
    double mvd_x = 0.0;
    double mvd_y = 0.0;
    for(LandmarkObs &obs: mapped_observations){
      double x_diff = obs.x - obsrvd_landmarks[obs.id].x;
      double y_diff = obs.y - obsrvd_landmarks[obs.id].y;
      mvd_x += x_diff*x_diff;
      mvd_y += y_diff*y_diff;
    }

    double std_x = 2*std_landmark[0]*std_landmark[0];
    double std_y = 2*std_landmark[1]*std_landmark[1];
    particle.weight = exp(-mvd_x/std_x - mvd_y/std_y);
  }

  // update weights
  weights = vector<double>(num_particles);
  for(int i=0; i< num_particles; ++i){
    weights[i] = particles[i].weight;
  }

}

void ParticleFilter::resample() {
  default_random_engine gen;

  // distribution based on particle weights
  discrete_distribution<int> dist_weight(weights.begin(), weights.end());

  vector<Particle> reSampled;
  for(int i=0; i< num_particles; i++){
    // pick a particle with replacement
    Particle particle = particles[dist_weight(gen)];
    reSampled.push_back(particle);
  }

  particles = reSampled;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
