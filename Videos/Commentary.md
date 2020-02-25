# DodgeAndHovers

The starship aims for the red waypoint and choses a dangerous path by pathing next to the planet. The ship successfully dodges the planet and gets close to the waypoint. However, it is not very precise and hovers around it because of the forward model error. 

# Hovers 

The starship aims for the red waypoint, hovers around it for a while but ends up getting caught by the planet's gravity. The starship is not precise enough with its actions, mostly because it has a high uncertainty due to model errors. Also crashing on a planet at the end of an episode hardly penalises the agent, which is why it can happend sometimes>

# oneDirection

The starship has to turn its boosters at full capacity to successfully avoid the giant planet's gravity. However, the optimizer was unable to turn the boosters down to stop at the red waypoint it encounters twice.

# spawnOnWP 

The starship starts next to the red waypoint, loses track of it and has to dodge the planet before coming back to hover around it.

