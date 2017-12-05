from __future__ import print_function
import mxnet as mx
from mxnet import nd, autograd
from mxnet import gluon
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from IPython import display
import gym
import math
from collections import namedtuple
import time
f = open('results.txt','w')


class Options:
    def __init__(self):
        #Articheture
        self.batch_size = 32 # The size of the batch to learn the Q-function
        self.image_size = 80 # Resize the raw input frame to square frame of size 80 by 80 
        
        #Trickes
        self.replay_buffer_size = 100000 # The size of replay buffer; set it to size of your memory (.5M for 50G available memory)
        self.learning_frequency = 4 # With Freq of 1/4 step update the Q-network
        self.skip_frame = 4 # Skip 4-1 raw frames between steps
        self.internal_skip_frame = 4 # Skip 4-1 raw frames between skipped frames
        self.frame_len = 4 # Each state is formed as a concatination 4 step frames [f(t-12),f(t-8),f(t-4),f(t)]
        self.Target_update = 10000 # Update the target network each 10000 steps
        self.epsilon_min = 0.01 # Minimum level of stochasticity of policy (epsilon)-greedy
        self.annealing_end = 4*self.replay_buffer_size # The number of step it take to linearly anneal the epsilon to it min value
        self.gamma = 0.99 # The discount factor
        self.replay_start_size = 500 # Start to backpropagated through the network, learning starts
        self.no_op_max = 30 / self.skip_frame # Run uniform policy for first 30 times step of the beginning of the game
        
        #GAN training
        self.batch_size_GAN = 32
        self.intrinsic_beta = .01
        self.input_variance_GAN = 7/255.
        self.d_pretrain_epochs_GAN = 0
        self.d_label_switch_prob_GAN = 5e-2
        self.innovation_score_threshold = .5
        self.reset_frequency_GAN = 800
        
        self.load_saved = False
        self.params_savefile = ''
        
        #otimization
        self.num_episode = 100000 # Number episode to run the algorithm
        self.lr = 0.00015 # RMSprop learning rate
        self.gamma1 = 0.95 # RMSprop gamma1
        self.gamma2 = 0.95 # RMSprop gamma2
        self.rms_eps = 0.01 # RMSprop epsilon bias
        self.ctx = mx.gpu() # Enables gpu if available, if not, set it to mx.cpu()
        
opt = Options()
env_name = 'MontezumaRevengeNoFrameskip-v4' # Set the desired environment
env = gym.make(env_name)
num_action = env.action_space.n # Extract the number of available action from the environment setting

manualSeed = 1 # random.randint(1, 10000) # Set the desired seed to reproduce the results
mx.random.seed(manualSeed)
attrs = vars(opt)
print (', '.join("%s: %s" % item for item in attrs.items()))

def get_dqn():
    DQN = gluon.nn.Sequential()
    with DQN.name_scope():
        #first layer
        DQN.add(gluon.nn.Conv2D(channels=32, kernel_size=8,strides = 4,padding = 0))
        DQN.add(gluon.nn.BatchNorm(axis = 1, momentum = 0.1,center=True))
        DQN.add(gluon.nn.Activation('relu'))
        #second layer
        DQN.add(gluon.nn.Conv2D(channels=64, kernel_size=4,strides = 2))
        DQN.add(gluon.nn.BatchNorm(axis = 1, momentum = 0.1,center=True))
        DQN.add(gluon.nn.Activation('relu'))
        #tird layer
        DQN.add(gluon.nn.Conv2D(channels=64, kernel_size=3,strides = 1))
        DQN.add(gluon.nn.BatchNorm(axis = 1, momentum = 0.1,center=True))
        DQN.add(gluon.nn.Activation('relu'))
        DQN.add(gluon.nn.Flatten())
        #fourth layer
        DQN.add(gluon.nn.Dense(512,activation ='relu'))
        #fifth layer
        DQN.add(gluon.nn.Dense(num_action,activation ='relu'))
    return DQN

dqn = get_dqn()
dqn.collect_params().initialize(mx.init.Normal(0.02), ctx=opt.ctx)
DQN_trainer = gluon.Trainer(dqn.collect_params(),'RMSProp', \
                          {'learning_rate': opt.lr ,'gamma1':opt.gamma1,'gamma2': opt.gamma2,'epsilon': opt.rms_eps,'centered' : True})
dqn.collect_params().zero_grad()
target_dqn = get_dqn()
target_dqn.collect_params().initialize(mx.init.Normal(0.02), ctx=opt.ctx)
    
noise_dim = 128
last_layer_dim = 6*6*128

def get_discriminator():
    Discriminator = gluon.nn.Sequential()
    with Discriminator.name_scope():
        #first layer
        Discriminator.add(gluon.nn.Conv2D(channels=32, kernel_size=8,strides = 4,padding = 0))
        Discriminator.add(gluon.nn.BatchNorm(axis = 1, momentum = 0.1,center=True))
        Discriminator.add(gluon.nn.LeakyReLU(.1))
        #second layer
        Discriminator.add(gluon.nn.Conv2D(channels=64, kernel_size=4,strides = 2))
        Discriminator.add(gluon.nn.BatchNorm(axis = 1, momentum = 0.1,center=True))
        Discriminator.add(gluon.nn.LeakyReLU(.1))
        #third layer
        Discriminator.add(gluon.nn.Conv2D(channels=64, kernel_size=3,strides = 1))
        Discriminator.add(gluon.nn.BatchNorm(axis = 1, momentum = 0.1,center=True))
        Discriminator.add(gluon.nn.LeakyReLU(.1))
        Discriminator.add(gluon.nn.Flatten())

        #fourth layer
        Discriminator.add(gluon.nn.Dense(int(noise_dim)))

        #fifth layer
        Discriminator.add(gluon.nn.Dense(1, activation='sigmoid'))
        Discriminator.add(gluon.nn.LeakyReLU(.1))
    #print(Discriminator)
    return Discriminator

class Reshape(gluon.nn.Block):
    def __init__(self, **kwargs):
        size = kwargs.pop("dims")
        super(Reshape, self).__init__(**kwargs)
        with self.name_scope():
            self.size = size

    def forward(self,x):
        return x.reshape((x.shape[0],)+(self.size))
        
def get_generator():
    
    Generator = gluon.nn.Sequential()
    with Generator.name_scope():
        #first layer
        Generator.add(gluon.nn.Dense(last_layer_dim,activation ='relu'))
        Generator.add(Reshape(dims=(128,6,6)))

        #second layer
        Generator.add(gluon.nn.Conv2DTranspose(channels=64, kernel_size=3,strides = 1))
        Generator.add(gluon.nn.BatchNorm(axis = 1, momentum = 0.1,center=True))
        Generator.add(gluon.nn.Activation('relu'))

        #third layer
        Generator.add(gluon.nn.Conv2DTranspose(channels=32, kernel_size=5,strides = 2))
        Generator.add(gluon.nn.BatchNorm(axis = 1, momentum = 0.1,center=True))
        Generator.add(gluon.nn.Activation('relu'))

        #fourth layer
        Generator.add(gluon.nn.Conv2DTranspose(channels=4, kernel_size=8,strides = 4,padding = 0))
        Generator.add(gluon.nn.BatchNorm(axis = 1, momentum = 0.1,center=True))
        Generator.add(gluon.nn.Activation('tanh'))
    return Generator

loss_bce= gluon.loss.SigmoidBinaryCrossEntropyLoss(batch_axis=0, from_sigmoid=True)
def StepGAN(batch_state, discriminator, discriminator_trainer, generator, generator_trainer, should_print = False, train_generator=True):
    generator.collect_params().zero_grad()
    discriminator.collect_params().zero_grad()
    loss_disc = 0.
    z = nd.random.normal(shape=(batch_state.shape[0],noise_dim), ctx=opt.ctx)
    batch_fake_states = generator(z)
    switch_var = nd.random.uniform(shape=(batch_state.shape[0]), ctx=opt.ctx) > opt.d_label_switch_prob_GAN
    noisy_one = nd.random.uniform(shape=(batch_state.shape[0]), low=1.,high=1., ctx=opt.ctx)
    noisy_zero = nd.random.uniform(shape=(batch_state.shape[0]), low=0.,high=.0, ctx=opt.ctx)
    noisy_one_labels = (switch_var) * noisy_one + (switch_var == 0) * noisy_zero
    noisy_zero_labels = (switch_var) * noisy_zero + (switch_var == 0) * noisy_one
    input_perturbation = opt.input_variance_GAN * nd.random.normal(shape=batch_state.shape, ctx=opt.ctx)
    batch_train_states = 1.8 * batch_state - 0.9 + input_perturbation
    with autograd.record():
        score_reals = discriminator(batch_train_states)
        score_fakes = discriminator(batch_fake_states)
        loss_discriminator_reals = loss_bce(score_reals,noisy_one_labels)
        loss_discriminator_fakes = loss_bce(score_fakes,noisy_zero_labels)
        loss_discriminator = nd.mean(loss_discriminator_reals) + nd.mean(loss_discriminator_fakes)
    loss_discriminator.backward()
    loss_disc = loss_discriminator.asnumpy()[0]
    discriminator_trainer.step(batch_state.shape[0])
    generator.collect_params().zero_grad()
    discriminator.collect_params().zero_grad()
    if should_print:
        show_gan(score_fakes[0].asscalar(), batch_fake_states[0,0], score_reals[0].asscalar(), batch_state[0,0])
    if train_generator:
        z = nd.random.normal(shape=(batch_state.shape[0],noise_dim), ctx=opt.ctx)
        with autograd.record():
            batch_fake_states = generator(z)
            score_generator = discriminator(batch_fake_states)
            loss_generator = nd.mean(loss_bce(score_generator,nd.ones(shape=(batch_state.shape[0]), ctx=opt.ctx)))
        loss_generator.backward()
        generator_trainer.step(batch_state.shape[0])
    return loss_disc

dlr_best = 0.000120
glr_best = 0.002626

def RunGAN(episode_states, discriminator, discriminator_trainer, generator, generator_trainer, should_print = False):
    sampler = mx.gluon.data.SequentialSampler(len(episode_states))
    batch_sampler = mx.gluon.data.BatchSampler(sampler, opt.batch_size_GAN)
    batch_state_GAN = nd.empty((opt.batch_size_GAN,opt.frame_len,opt.image_size,opt.image_size), opt.ctx)
    loss_disc = 0.
    for batch_indices in batch_sampler:
        for i in range(len(batch_indices)):
            index = batch_indices[i]
            batch_state_GAN[i] = nd.array(episode_states[index],opt.ctx)
        loss_disc = StepGAN(batch_state_GAN, discriminator, discriminator_trainer, generator, generator_trainer, False)

    results = 'Episode loss %f'%(loss_disc)
    if should_print:
        print(results)
    
Transition = namedtuple('Transition',('state', 'action', 'next_state', 'reward','done'))
class Replay_Buffer():
    def __init__(self, replay_buffer_size):
        self.replay_buffer_size = replay_buffer_size
        self.memory = []
        self.position = 0
    def push(self, *args):
        if len(self.memory) < self.replay_buffer_size:
            self.memory.append(None)
            
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.replay_buffer_size
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    
def preprocess(raw_frame, currentState = None, initial_state = False):
    raw_frame = nd.array(raw_frame,mx.cpu())
    raw_frame = nd.reshape(nd.mean(raw_frame, axis = 2),shape = (raw_frame.shape[0],raw_frame.shape[1],1))
    raw_frame = mx.image.imresize(raw_frame,  opt.image_size, opt.image_size)
    raw_frame = nd.transpose(raw_frame, (2,0,1))
    raw_frame = raw_frame.astype('float32')/255.
    if initial_state == True:
        state = raw_frame
        for _ in range(opt.frame_len-1):
            state = nd.concat(state , raw_frame, dim = 0)
    else:
        state = mx.nd.concat(currentState[1:,:,:], raw_frame, dim = 0)
    return state

def rew_clipper(rew):
    if rew>0.:
        return 1.
    elif rew<0.:
        return -1.
    else:
        return 0

def renderimage(next_frame):
    if render_image:
        plt.imshow(next_frame)
        plt.show()
        #display.clear_output(wait=True)
        #time.sleep(.1)
        
def renderimage_preprocess(next_frame):
    if render_image:
        state = 255.*(preprocess(next_frame, initial_state = True)[0])
        preprocessed_frame = state.astype('uint8').asnumpy()
        plt.imshow(preprocessed_frame)
        plt.show()
        #display.clear_output(wait=True)
        #time.sleep(.1)
        
def show_gan(score_fake, frame_fake, score_real, frame_real):
    print_frame_fake = (255.*(frame_fake+1.)/2.).astype('uint8').asnumpy()
    print_frame_real = (255.*frame_real).astype('uint8').asnumpy()
    fig = plt.figure()
    a=fig.add_subplot(1,2,1)
    imgplot = plt.imshow(print_frame_real)
    a.set_title("Real "+str(score_real))

    a2=fig.add_subplot(1,2,2)
    imgplot2 = plt.imshow(print_frame_fake)
    a2.set_title("Fake "+str(score_fake))
    plt.show()
    #display.clear_output(wait=True)
    #time.sleep(.1)
        
def save_frame(state, file_number):
    preprocessed_frame = (255.*state[0]).astype('uint8').asnumpy()
    fig = plt.figure()
    plt.imshow(preprocessed_frame)
    plt.show()
    #fig.savefig('frame_saves/'+str(file_number)+'.png')
    display.clear_output(wait=True)
    time.sleep(.1)
    plt.close(fig)
l2loss = gluon.loss.L2Loss(batch_axis=0)


D = get_discriminator()
D.collect_params().initialize(mx.init.Normal(0.02), ctx=opt.ctx)
G = get_generator()
G.collect_params().initialize(mx.init.Normal(0.02), ctx=opt.ctx)
D_trainer = gluon.Trainer(D.collect_params(),'Adam', {'learning_rate': dlr_best})
D.collect_params().zero_grad()
G_trainer = gluon.Trainer(G.collect_params(),'Adam', {'learning_rate': glr_best})
G.collect_params().zero_grad()

frame_counter = 0 # Counts the number of steps so far
saved_frame_counter = 0
annealing_count = 0. # Counts the number of annealing steps
epis_count = 0. # Counts the number episodes so far
replay_memory = Replay_Buffer(opt.replay_buffer_size) # Initialize the replay buffer
tot_clipped_reward = np.zeros(opt.num_episode) 
saved_frame_counter = 0
tot_reward = np.zeros(opt.num_episode)
moving_average_clipped = 0.
moving_average = 0.
episode_states = []
tot_intrinsic_reward = np.zeros(opt.num_episode)

if opt.load_saved:
    dqn.load_params(opt.params_savefile,opt.ctx)

batch_state = nd.empty((opt.batch_size,opt.frame_len,opt.image_size,opt.image_size), opt.ctx)
batch_state_next = nd.empty((opt.batch_size,opt.frame_len,opt.image_size,opt.image_size), opt.ctx)
for i in range(opt.num_episode):
    cum_clipped_reward = 0
    intrinsic_reward = 0.
    cum_reward = 0
    next_frame = env.reset()
    state = preprocess(next_frame, initial_state = True)
    t = 0
    done = False
    
    if epis_count % opt.reset_frequency_GAN == 0:
            D = get_discriminator()
            D.collect_params().initialize(mx.init.Normal(0.02), ctx=opt.ctx)
            G = get_generator()
            G.collect_params().initialize(mx.init.Normal(0.02), ctx=opt.ctx)
            D_trainer = gluon.Trainer(D.collect_params(),'Adam', {'learning_rate': dlr_best})
            D.collect_params().zero_grad()
            G_trainer = gluon.Trainer(G.collect_params(),'Adam', {'learning_rate': glr_best})
            G.collect_params().zero_grad()

    while not done:
        previous_state = state
        # show the frame
        sample = random.random()
        if frame_counter > opt.replay_start_size:
            annealing_count += 1
        if frame_counter == opt.replay_start_size:
            print('DQN annealing and learning are started ')
            
        eps = np.maximum(1.-annealing_count/opt.annealing_end,opt.epsilon_min)
        effective_eps = eps
        if t < opt.no_op_max:
            effective_eps = 1.
        
        # epsilon greedy policy
        if sample < effective_eps:
            action = random.randint(0, num_action - 1)
        else:
            data = nd.array(state.reshape([1,opt.frame_len,opt.image_size,opt.image_size]),opt.ctx)
            action = int(nd.argmax(dqn(data),axis=1).as_in_context(mx.cpu()).asscalar())
        
        # Skip frame
        rew = 0
        for skip in range(opt.skip_frame-1):
            next_frame, reward, done,_ = env.step(action)
            cum_clipped_reward += rew_clipper(reward)
            rew += reward
            for internal_skip in range(opt.internal_skip_frame-1):
                _ , reward, done,_ = env.step(action)
                cum_clipped_reward += rew_clipper(reward)
                rew += reward
                
        next_frame_new, reward, done, _ = env.step(action)
        cum_clipped_reward += rew_clipper(reward)
        rew += reward
        cum_reward += rew
        
        # Reward clipping
        reward = rew_clipper(rew)
        next_frame = np.maximum(next_frame_new,next_frame)
        state = preprocess(next_frame, state)
        episode_states.append(state)
        replay_memory.push((previous_state*255.).astype('uint8'),action,(state*255.).astype('uint8'),reward,done)
            
            
        if frame_counter > opt.replay_start_size:
            if frame_counter % opt.learning_frequency == 0:
                transitions = replay_memory.sample(opt.batch_size)
                batch = Transition(*zip(*transitions))
                for j in range(opt.batch_size):
                    batch_state[j] = nd.array(batch.state[j],opt.ctx).astype('float32')/255.
                    batch_state_next[j] = nd.array(batch.next_state[j],opt.ctx).astype('float32')/255.
                
                batch_discriminator_state = 1.8 * batch_state - 0.9
                with autograd.train_mode():
                    score_reals = D(batch_discriminator_state)
                disc_scores = score_reals
                innovation_rewards = 1. - disc_scores
                batch_reward = nd.array(batch.reward,opt.ctx) + opt.intrinsic_beta * innovation_rewards
                intrinsic_reward = intrinsic_reward + nd.sum(innovation_rewards).asscalar()
                batch_action = nd.array(batch.action,opt.ctx).astype('uint8')
                batch_done = nd.array(batch.done,opt.ctx)
                with autograd.record():
                    Q_sp = nd.max(target_dqn(batch_state_next),axis = 1)
                    Q_sp = Q_sp*(nd.ones(opt.batch_size,ctx = opt.ctx)-batch_done)
                    Q_s_array = dqn(batch_state)
                    Q_s = nd.pick(Q_s_array,batch_action,1)
                    loss = nd.mean(l2loss(Q_s ,  (batch_reward + opt.gamma *Q_sp)))
                loss.backward()
        
        t += 1
        frame_counter += 1
        
        # Save the model and update Target model
        if frame_counter > opt.replay_start_size:
            if frame_counter % opt.Target_update == 0 :
                check_point = frame_counter / (opt.Target_update *100)
                fdqn = './target_%s_%d' % (env_name,int(check_point))
                dqn.save_params(fdqn)
                target_dqn.load_params(fdqn, opt.ctx)
        if done:
            if epis_count % 10. == 0. :
                results = 'epis[%d],eps[%f],durat[%d],fnum=%d, cum_cl_rew = %d, cum_rew = %d,tot_cl = %d , tot = %d, tot_intrinsic = %f'\
                  %(epis_count,eps,t+1,frame_counter,cum_clipped_reward,cum_reward,moving_average_clipped,moving_average, intrinsic_reward)
                print(results)
                f.write('\n' + results)
                
    # Replay GAN on last episode
    if epis_count % 10 == 0:
        RunGAN(episode_states, D, D_trainer, G, G_trainer, True)
        episode_states = []
    
    
    epis_count += 1
    tot_clipped_reward[int(epis_count)-1] = cum_clipped_reward
    tot_intrinsic_reward[int(epis_count)-1] = intrinsic_reward
    tot_reward[int(epis_count)-1] = cum_reward
    if epis_count > 50.:
        moving_average_clipped = np.mean(tot_clipped_reward[int(epis_count)-1-50:int(epis_count)-1])
        moving_average = np.mean(tot_reward[int(epis_count)-1-50:int(epis_count)-1])
f.close()
from tempfile import TemporaryFile
outfile = TemporaryFile()
outfile_clip = TemporaryFile()
outfile_intrinsic = TemporaryFile()
np.save(outfile, moving_average)
np.save(outfile_clip, moving_average_clipped)
np.save(outfile_intrinsic, tot_intrinsic_reward)

bandwidth = 1000 # Moving average bandwidth
total_clipped = np.zeros(int(epis_count)-bandwidth)
total_rew = np.zeros(int(epis_count)-bandwidth)
for i in range(int(epis_count)-bandwidth):
    total_clipped[i] = np.sum(tot_clipped_reward[i:i+bandwidth])/bandwidth
    total_rew[i] = np.sum(tot_reward[i:i+bandwidth])/bandwidth
t = np.arange(int(epis_count)-bandwidth)
belplt = plt.plot(t,total_rew[0:int(epis_count)-bandwidth],"r", label = "Return")
plt.legend()#handles[likplt,belplt])
print('Running after %d number of episodes' %epis_count)
plt.xlabel("Number of episode")
plt.ylabel("Average Reward per episode")
plt.show()
likplt = plt.plot(t,total_clipped[0:opt.num_episode-bandwidth],"b", label = "Clipped Return")
plt.legend()#handles[likplt,belplt])
plt.xlabel("Number of episode")
plt.ylabel("Average clipped Reward per episode")
plt.show()


