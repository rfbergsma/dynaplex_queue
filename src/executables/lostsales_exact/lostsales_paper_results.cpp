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
	config.Set("leadtime", 3);
	config.Set("demand_dist", DynaPlex::VarGroup({
		{"type", "geometric"},
		{"mean", 5.0}
		}));

	DynaPlex::MDP mdp = dp.GetMDP(config);
	auto policy = mdp->GetPolicy("base_stock");
}