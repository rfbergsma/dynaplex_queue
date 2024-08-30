#include <iostream>
#include "dynaplex/dynaplexprovider.h"

using namespace DynaPlex;

int main() {

	auto& dp = DynaPlexProvider::Get();
	
	DynaPlex::VarGroup config;
	//retrieve MDP registered under the id string "lost_sales":
	config.Add("id", "lost_sales");
	config.Add("h", 1.0);
	config.Set("p", 9.0);
	config.Set("leadtime", 2);
	config.Set("demand_dist", DynaPlex::VarGroup({
		{"type", "geometric"},
		{"mean", 5.0}
		}));
	try {
		DynaPlex::MDP mdp = dp.GetMDP(config);
		auto policy = mdp->GetPolicy("base_stock");
		auto solver = dp.GetExactSolver(mdp, { {"silent",false }, {"max_states",1000000} });
		double d = solver.ComputeCosts(policy);
		std::cout << "base_stock: " << d << std::endl;
		double d2 = solver.ComputeCosts();
		std::cout << "optimal   : " << d2 << std::endl;
	}
	catch (DynaPlex::Error& e)
	{
		std::cout<< e.what() <<std::endl;
	}

}