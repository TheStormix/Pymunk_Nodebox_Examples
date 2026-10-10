#encoding: utf-8
from __future__ import division
from nodebox.graphics import *
import pymunk
import pymunk.pyglet_util
import random, math
from rl_agent import QLearningAgent

space = pymunk.Space()

def createBody(x,y,shape,*shapeArgs):
    body = pymunk.Body()
    body.position = x, y
    s = shape(body, *shapeArgs)
    s.mass = 1
    s.friction = 1
    space.add(body, s)
    return s #shape!!!

s0=createBody(300, 300, pymunk.Poly, ((-20,-5),(-20,5),(20,15),(20,-15)))
s0.score=0
s3=createBody(200, 300, pymunk.Poly, ((-20,-5),(-20,5),(20,15),(20,-15)))
s3.color = (0, 255, 0, 255)
s3.score=0
# Tabular Q-learning policy for the green robot.
s3.body.action = 0
s3.body.previous_score = 0
agent = QLearningAgent(states=4, actions=2, alpha=0.30, gamma=0.90,
                       epsilon=0.60, epsilon_min=0.05, epsilon_decay=0.999,
                       seed=9)

s1=createBody(300, 200, pymunk.Circle, 10, (0,0))
S2=[]
for i in range(1):
    s2=createBody(350, 250, pymunk.Circle, 10, (0,0))
    s2.color = (255, 0, 0, 255)
    S2.append(s2)


def getAngle(x,y,x1,y1):
    return math.atan2(y1-y, x1-x)

def getDist(x,y,x1,y1):
    return ((x-x1)**2+(y-y1)**2)**0.5

def inCircle(x,y,cx,cy,R):
    if (x-cx)**2+(y-cy)**2 < R**2:
        return True
    return False

def inSector(x, y, cx, cy, R, a):
    # Angular distance must wrap correctly at -pi/pi and 0/2pi.
    if not inCircle(x, y, cx, cy, R):
        return False
    angle = getAngle(cx, cy, x, y)
    angular_difference = (angle-a+math.pi) % (2*math.pi) - math.pi
    return abs(angular_difference) < 0.5

def strategy(b=s3.body):
    v=100
    a=b.angle
    b.velocity=v*cos(a), v*sin(a)
    x,y=b.position
    R=getDist(x,y,350,250)
    #print R, b.angle
    line(x,y,*s1.body.position,stroke=Color(0))
    if canvas.frame%100==0:
        if R>180:
            b.angle=getAngle(x,y,350,250)
        else:
            b.angle=getAngle(x,y,*s1.body.position) #2*math.pi*random.random()

def strategy2(b=s3.body):
    u"""Green robot policy with epsilon-greedy tabular Q-learning.

    Observe every 10 frames. Previous (state, action) is updated with
    reward and the next state's best value (Bellman TD target).
    """
    speed = 100
    heading = b.angle
    b.velocity = speed * math.cos(heading), speed * math.sin(heading)
    x, y = b.position
    radius = getDist(x, y, 350, 250)

    # Visualize the 100-pixel, +/-0.5 rad ultrasonic sensor sector.
    ellipse(x, y, 200, 200, stroke=Color(0.5))
    line(x, y, x+100*math.cos(heading+0.5),
         y+100*math.sin(heading+0.5), stroke=Color(0.5))
    line(x, y, x+100*math.cos(heading-0.5),
         y+100*math.sin(heading-0.5), stroke=Color(0.5))

    if canvas.frame % 10 == 0:
        sees_target = inSector(s1.body.position[0], s1.body.position[1],
                               x, y, 100, heading)
        sees_anti = any(inSector(item.body.position[0], item.body.position[1],
                                 x, y, 100, heading) for item in S2)
        # Distance to arena centre has priority: safe boundary override.
        if radius > 160:
            state = 3
        elif sees_anti:
            state = 2
        elif sees_target:
            state = 1
        else:
            state = 0

        # Reward is observed after the previous decision.
        score_change = s3.score - b.previous_score
        reward = 4.0 * score_change
        b.previous_score = s3.score
        if agent.previous_action is not None:
            if state == 1:
                reward += 0.4 if agent.previous_action == 0 else -0.2
            elif state == 2:
                reward += -0.4 if agent.previous_action == 0 else 0.4
            elif state == 3:
                reward -= 0.6

        b.action = agent.step(state, reward)
        if b.action == 1:
            if state == 2:
                # Turn away when an anti-target is seen.
                b.angle = (heading + math.pi) % (2*math.pi)
            else:
                b.angle = 2*math.pi*random.random()

        if agent.steps % 100 == 0:
            print('RL decisions=%d epsilon=%.3f Q=%s' %
                  (agent.steps, agent.epsilon, agent.q))

    # Physics boundary override (original simulation uses radius 180).
    if radius > 165:
        b.angle = getAngle(x, y, 350, 250)

def scr(s,s0,s3,p=1):
    bx,by=s.body.position
    s0x,s0y=s0.body.position
    s3x,s3y=s3.body.position
    if not inCircle(bx,by,350,250,180):
        if getDist(bx,by,s0x,s0y)<getDist(bx,by,s3x,s3y):
            s0.score=s0.score+p
        else:
            s3.score=s3.score+p
        s.body.position=random.randint(200,400),random.randint(200,300)

def score():
    u"""визначає переможця"""
    scr(s1,s0,s3)
    for s in S2:
        scr(s,s0,s3,p=-1)


def manualControl():
    u"""Керування роботом з мишки або клавіатури"""
    v=10 # швидкість
    b=s0.body
    a=b.angle
    x,y=b.position
    vx,vy=b.velocity
    if canvas.keys.char=="a":
        b.angle-=0.1
    if canvas.keys.char=="d":
        b.angle+=0.1
    if canvas.keys.char=="w":
        b.velocity=vx+v*cos(a), vy+v*sin(a)
    if canvas.mouse.button==LEFT:
        b.angle=getAngle(x,y,*canvas.mouse.xy)
        b.velocity=vx+v*cos(a), vy+v*sin(a)

def simFriction():
    for s in [s0,s1,s3]+S2:
        s.body.velocity=s.body.velocity[0]*0.9, s.body.velocity[1]*0.9
        s.body.angular_velocity=s.body.angular_velocity*0.9
        #s.body.update_velocity(s.body, (0.,0.), 0.9, 0.02)

draw_options = pymunk.pyglet_util.DrawOptions()

def draw(canvas):
    canvas.clear()
    fill(0,0,0,1)
    text("%i %i"%(s0.score,s3.score),20,20)
    text("RL: decisions=%d epsilon=%.2f" % (agent.steps, agent.epsilon), 20, 42)
    nofill()
    ellipse(350, 250, 350, 350, stroke=Color(0))
    manualControl()
    strategy2()
    score()
    simFriction()
    space.step(0.02)
    space.debug_draw(draw_options)

canvas.size = 700, 500
canvas.run(draw)
