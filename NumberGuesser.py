import pygame
import numpy as np
#import cupy as cp
from PIL import Image
from skimage import data
import cv2
from skimage.transform import resize
import os

rescale_factor = 60
newSurface = pygame.Surface((500, 100))
newSurface.fill((200, 200, 200))
rect = pygame.Rect(0, 0, 500, 500)

class NumberGuesser:

	def __init__(self, net):

		self.net = net
		self.screen = pygame.display.set_mode((500, 600))
		self.screen.fill((255, 255, 255))
		self.brush = pygame.image.load("brush.png")
		self.running = True
		self.brush = pygame.transform.scale(self.brush, (rescale_factor, rescale_factor))
		self.run()
		
	def update(self, events, font, sub):
		for event in events:

			if pygame.mouse.get_pressed()[0]:

				try:
					x, y = event.pos
					self.screen.blit(self.brush, (x - rescale_factor/2, y - rescale_factor/2))
					pygame.display.update()
				except AttributeError:
					pass

			if event.type == pygame.KEYDOWN:
					
				number_guessed = font.render('Guess: ', True, (0, 0, 0))
				confidence = font.render('Confidence: ', True, (0, 0, 0))

				self.screen.fill((255, 255, 255))
				self.screen.blit(newSurface, (0, 500))
				self.screen.blit(number_guessed, (20, 510))
				self.screen.blit(confidence, (20, 560))
				pygame.display.update()

			if event.type == pygame.MOUSEBUTTONUP:

				pygame.image.save(sub, "num.png")
				im = cv2.imread("num.png")

				# Invert pixel values
				im = cv2.bitwise_not(im)
				# Combine channels to grey scale
				im = np.dot(im[...,:3], [1, 0.0, 0.0])
				im = resize(im, (28, 28), anti_aliasing_sigma=4, clip = True)
				im = im/127.5 - 1.0
				im = np.reshape(im, (1, 784))
				im = np.array(im)[0]
			
				num = self.net.feedforward(im)

				self.screen.blit(newSurface, (0, 500))

				number_guessed = font.render('Guess: %d' % np.argmax(num), True, (0, 0, 0))
				confidence_guess = float(np.max(num)*100)
				confidence = font.render(f'Confidence: {confidence_guess:.2f}%', True, (0, 0, 0))
					
				self.screen.blit(number_guessed, (20, 510))
				self.screen.blit(confidence, (20, 560))

				pygame.display.update()

	def run(self):

		pygame.init()
		pygame.font.init()

		font = pygame.font.SysFont('Arial', 30)
		pygame.display.set_caption('Number Guesser')

		number_guessed = font.render('Guess: ', True, (0, 0, 0))
		confidence = font.render('Confidence: ', True, (0, 0, 0))

		self.screen.blit(newSurface, (0, 500))
		self.screen.blit(number_guessed, (20, 510))
		self.screen.blit(confidence, (20, 560))

		pos1 = (0, 32)

	
		clock = pygame.time.Clock()
		sub = self.screen.subsurface(rect)
		pygame.display.update()

		while self.running:


			events = pygame.event.get()

			for e in events:

				if e.type == pygame.QUIT:

					self.running = False
			
			self.update(events, font, sub)

		try:
			os.remove('num.png')
		except FileNotFoundError:
			pass
