# -*- coding: utf-8 -*-

"""
@date: 2020/8/29 上午10:57
@file: ucf101.py
@author: zj
@description: 
"""

import os

from .base_dataset import VideoRecord, BaseDataset

classes = ['ApplyEyeMakeup', 'ApplyLipstick', 'Archery', 'BabyCrawling',
           'BalanceBeam', 'BandMarching', 'BaseballPitch', 'Basketball',
           'BasketballDunk', 'BenchPress', 'Biking', 'Billiards',
           'BlowDryHair', 'BlowingCandles', 'BodyWeightSquats', 'Bowling',
           'BoxingPunchingBag', 'BoxingSpeedBag', 'BreastStroke',
           'BrushingTeeth', 'CleanAndJerk', 'CliffDiving', 'CricketBowling',
           'CricketShot', 'CuttingInKitchen', 'Diving', 'Drumming', 'Fencing',
           'FieldHockeyPenalty', 'FloorGymnastics', 'FrisbeeCatch',
           'FrontCrawl', 'GolfSwing', 'Haircut', 'Hammering', 'HammerThrow',
           'HandstandPushups', 'HandstandWalking', 'HeadMassage', 'HighJump',
           'HorseRace', 'HorseRiding', 'HulaHoop', 'IceDancing',
           'JavelinThrow', 'JugglingBalls', 'JumpingJack', 'JumpRope',
           'Kayaking', 'Knitting', 'LongJump', 'Lunges', 'MilitaryParade',
           'Mixing', 'MoppingFloor', 'Nunchucks', 'ParallelBars',
           'PizzaTossing', 'PlayingCello', 'PlayingDaf', 'PlayingDhol',
           'PlayingFlute', 'PlayingGuitar', 'PlayingPiano', 'PlayingSitar',
           'PlayingTabla', 'PlayingViolin', 'PoleVault', 'PommelHorse',
           'PullUps', 'Punch', 'PushUps', 'Rafting', 'RockClimbingIndoor',
           'RopeClimbing', 'Rowing', 'SalsaSpin', 'ShavingBeard', 'Shotput',
           'SkateBoarding', 'Skiing', 'Skijet', 'SkyDiving', 'SoccerJuggling',
           'SoccerPenalty', 'StillRings', 'SumoWrestling', 'Surfing', 'Swing',
           'TableTennisShot', 'TaiChi', 'TennisSwing', 'ThrowDiscus',
           'TrampolineJumping', 'Typing', 'UnevenBars', 'VolleyballSpiking',
           'WalkingWithDog', 'WallPushups', 'WritingOnBoard', 'YoYo']


class UCF101(BaseDataset):

    def __init__(self,
                 *args,
                 split=1,
                 **kwargs):
        assert isinstance(split, int) and split in (1, 2, 3)

        self.split = split
        super(UCF101, self).__init__(*args, **kwargs)
        self.base_index = 0
        self.img_prefix = 'img_'

    def _update_video(self, annotation_dir, is_train=True):
        if is_train:
            annotation_path = os.path.join(annotation_dir, f'ucf101_train_split_{self.split}_rawframes.txt')
        else:
            annotation_path = os.path.join(annotation_dir, f'ucf101_val_split_{self.split}_rawframes.txt')

        if not os.path.isfile(annotation_path):
            raise ValueError(f'{annotation_path}不是文件路径')

        self.video_list = [VideoRecord(x.strip().split(' ')) for x in open(annotation_path)]

    def _update_class(self):
        self.classes = classes
