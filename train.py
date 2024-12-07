import numpy as np
import os
from DDQN_discrete import *

if __name__ == '__main__':
    os.system('mkdir -p saved-single && mkdir -p saved-double')

    # DQN
    Q_1, Q_2, performance, episode_rewards = main(double=False)

    np.save('saved-single/episode-rewards.npy', episode_rewards)

    torch.save(Q_1.state_dict(), 'saved-single/q1.pt')
    torch.save(Q_2.state_dict(), 'saved-single/q2.pt')

    # DDQN
    Q_1, Q_2, performance, episode_rewards = main()

    np.save('saved-double/episode-rewards.npy', episode_rewards)

    torch.save(Q_1.state_dict(), 'saved-double/q1.pt')
    torch.save(Q_2.state_dict(), 'saved-double/q2.pt')

