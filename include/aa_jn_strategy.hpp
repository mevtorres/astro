#ifndef ASTRO_ACCELERATE_AA_JN_STRATEGY_HPP
#define ASTRO_ACCELERATE_AA_JN_STRATEGY_HPP

#include "aa_jn_plan.hpp"
#include "aa_strategy.hpp"

namespace astroaccelerate{


class aa_jn_strategy : public aa_strategy{
	public:
		aa_jn_strategy(){} 
		aa_jn_strategy(const aa_jn_plan &plan):my_val(plan.val()){}	
		bool ready() const{
			return true;
		}
		std::string name() const{
			return "JN_STRATEGY";
		}
		bool setup() {
			return true;
		}
	private:
		float my_val;
};

}
#endif
