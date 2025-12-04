#from .my_agent import MyNDaysNCampaignsAgent
#from .agent2 import TrialNDaysNCampaignsAgent
# from .agent4 import MyNDaysNCampaignsAgent
#from .agent10 import BigBuddyNDaysNCampaignsAgent as Agent10
from .agent11 import HybridRLAgent 

################### ACTUAL SUBMISSION #####################
#agent_submission = Agent10(name="brawl_stars")
agent_submission = HybridRLAgent(name="brawl_stars")
#agent_submission = TrialNDaysNCampaignsAgent(name="brawl_stars", shade_param=0.4)
###########################################################
