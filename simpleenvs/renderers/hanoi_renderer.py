import pygame
import distinctipy


WIDTH = 800
HEIGHT = 600

BACKGROUND_COLOUR = (255, 255, 255)
POLE_COLOUR = (128, 128, 128)

POLE_WIDTH = 8
POLE_PADDING = 16
POLE_HEIGHT = HEIGHT // 2
POLE_Y = HEIGHT // 4


class HanoiRenderer(object):
    def __init__(self, num_poles, num_disks):
        self.num_poles = num_poles
        self.num_disks = num_disks

        self._calculate_dimensions()

        self.pole_spacing = WIDTH // (num_poles + 1)

        self.disk_colours = [distinctipy.get_rgb256(colour) for colour in distinctipy.get_colors(self.num_disks)]

        # Initialise pygame and display window.
        pygame.init()
        self.display_window = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()

    def update(self, state):
        self.clock.tick(165)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()

        self.display_window.fill(BACKGROUND_COLOUR)

        # Draw poles.
        for pole_index in range(self.num_poles):
            pole_x = self.pole_spacing * (pole_index + 1)
            self._draw_pole(pole_x)

        # Draw disks.
        for pole_index in range(self.num_poles):
            disks_on_pole = [i for i, pos in enumerate(state) if pos == pole_index]
            pole_x = self.pole_spacing * (pole_index + 1)
            self._draw_disks(pole_x, disks_on_pole)

        pygame.display.update()

    def close(self):
        pygame.quit()

    def _calculate_dimensions(self):
        total_spacing = WIDTH * 0.8
        self.pole_spacing = total_spacing // (self.num_poles + 1)
        self.pole_start_x = (WIDTH - total_spacing) / 2

        self.disk_height = HEIGHT // 2 // (self.num_disks + 1)

        max_disk_width = self.pole_spacing - POLE_PADDING
        self.disk_widths = [
            int(max_disk_width * (disk_index + 1) / self.num_disks) for disk_index in range(self.num_disks)
        ]

    def _draw_pole(self, pole_x):
        pygame.draw.rect(self.display_window, POLE_COLOUR, (pole_x - POLE_WIDTH // 2, POLE_Y, POLE_WIDTH, POLE_HEIGHT))

    def _draw_disks(self, pole_x, disks_on_pole):
        for i, disk in enumerate(sorted(disks_on_pole, reverse=False)):
            disk_width = self.disk_widths[disk]
            disk_x = pole_x - disk_width // 2
            disk_y = POLE_Y + POLE_HEIGHT - (len(disks_on_pole) - i) * self.disk_height

            disk_colour = self.disk_colours[disk % len(self.disk_colours)]
            pygame.draw.rect(
                self.display_window,
                disk_colour,
                (disk_x, disk_y, disk_width, self.disk_height),
            )
