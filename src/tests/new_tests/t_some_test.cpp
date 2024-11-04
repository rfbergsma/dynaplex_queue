#include <gtest/gtest.h>
#include "dynaplex/dynaplexprovider.h"
using namespace DynaPlex;

namespace DynaPlex::Tests {    
	TEST(ExactAlgorithm, exact_test_case) {
		auto& dp = DynaPlexProvider::Get();

		DynaPlex::VarGroup config;
		//retrieve MDP registered under the id string "lost_sales":
		config.Add("id", "exact_test_case");

		DynaPlex::MDP mdp;
		DynaPlex::Policy policy;
		DynaPlex::VarGroup exact_config = DynaPlex::VarGroup{ {"max_states",100000}, {"silent", false } };

		ASSERT_NO_THROW({ mdp = dp.GetMDP(config); });
		ASSERT_NO_THROW({ policy = mdp->GetPolicy("greedy"); });
		{
			auto ExactSolver = dp.GetExactSolver(mdp, exact_config);
			double bs_costs, opt_costs;
			//ASSERT_NO_THROW({ bs_costs = ExactSolver.ComputeCosts(policy); });
			ASSERT_NO_THROW({ opt_costs = ExactSolver.ComputeCosts(true, policy);});

			//EXPECT_NEAR(bs_costs, 3, 0.005);
			EXPECT_NEAR(opt_costs, 20.0, 0.005);
		}

	}
}