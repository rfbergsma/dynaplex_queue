#include "dynaplex/dynaplexprovider.h"
#include "dynaplex/modelling/queue.h"
#include <vector>
#include <sstream>
#include <variant>
#include <numeric>
#include <fstream>
#include <string>
#include <iomanip>
#include <random>
#include <cmath>
#include "../../../lib/models/models/queue_mdp/mdp.h"
#include <cassert>


using namespace DynaPlex;
using DynaPlex::VarGroup;
using DynaPlex::DynaPlexProvider;
using DynaPlex::Models::queue_mdp::MDP;
using DynaPlex::RNG;



double expected_next_fil(int i, double lambda, double gamma) {
	if (i <= 0) return 0.0;
	double denom = lambda + gamma;
	if (denom <= 0.0) return 0.0;
	double alpha = lambda / denom;
	double beta = gamma / denom;

	double EX = 0.0;
	for (int j = 1; j <= i; ++j) {
		double p = std::pow(beta, i - j) * alpha;
		EX += j * p;
	}
	return EX;
}


int main() {

	auto& dp = DynaPlexProvider::Get();


	DynaPlex::VarGroup config;
	//retrieve MDP registered under the id string "lost_sales":
	auto path_to_json = dp.FilePath({ "mdp_config_examples", "queue_mdp" }, "mdp_config_0.json");
	//also possible to use a standard configuration file:
	config = VarGroup::LoadFromFile(path_to_json);

	//MDP mdp(config);


	// Optional: read id field if present
	if (config.HasKey("id")) {
		std::string id;
		config.Get("id", id);
		std::cout << "MDP id: " << id << "\n";
	}

	try
	{
		//DynaPlex::MDP mdp = dp.GetMDP(config);
	}
	catch (const std::exception& e)
	{
		std::cout << "exception: " << e.what() << std::endl;
	}
	return 0;
}
