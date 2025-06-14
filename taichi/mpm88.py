# MPM-MLS in 88 lines of Taichi code, originally created by @yuanming-hu
import taichi as ti
import random
import argparse


@ti.data_oriented
class MPM88:
    def __init__(self, n_particles):
        self.n_particles = n_particles

        self.n_grid = 128
        self.dx = 1 / self.n_grid
        self.dt = 1e-4

        self.p_rho = 1
        self.p_vol = (self.dx * 0.5) ** 2
        self.p_mass = self.p_vol * self.p_rho
        self.gravity = 9.8
        self.bound = 3
        self.E = 400

        self.x = ti.Vector.field(2, float, n_particles)
        self.v = ti.Vector.field(2, float, n_particles)
        self.C = ti.Matrix.field(2, 2, float, n_particles)
        self.J = ti.field(float, n_particles)

        self.grid_v = ti.Vector.field(2, float, (self.n_grid, self.n_grid))
        self.grid_m = ti.field(float, (self.n_grid, self.n_grid))

        self.init()

    @ti.kernel
    def substep(self):
        for i, j in self.grid_m:
            self.grid_v[i, j] = [0, 0]
            self.grid_m[i, j] = 0
        for p in self.x:
            Xp = self.x[p] / self.dx
            base = int(Xp - 0.5)
            fx = Xp - base
            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
            stress = -self.dt * 4 * self.E * self.p_vol * (self.J[p] - 1) / self.dx**2
            affine = ti.Matrix([[stress, 0], [0, stress]]) + self.p_mass * self.C[p]
            for i, j in ti.static(ti.ndrange(3, 3)):
                offset = ti.Vector([i, j])
                dpos = (offset - fx) * self.dx
                weight = w[i].x * w[j].y
                self.grid_v[base + offset] += weight * (
                    self.p_mass * self.v[p] + affine @ dpos
                )
                self.grid_m[base + offset] += weight * self.p_mass
        for i, j in self.grid_m:
            if self.grid_m[i, j] > 0:
                self.grid_v[i, j] /= self.grid_m[i, j]
            self.grid_v[i, j].y -= self.dt * self.gravity
            if i < self.bound and self.grid_v[i, j].x < 0:
                self.grid_v[i, j].x = 0
            if i > self.n_grid - self.bound and self.grid_v[i, j].x > 0:
                self.grid_v[i, j].x = 0
            if j < self.bound and self.grid_v[i, j].y < 0:
                self.grid_v[i, j].y = 0
            if j > self.n_grid - self.bound and self.grid_v[i, j].y > 0:
                self.grid_v[i, j].y = 0
        for p in self.x:
            Xp = self.x[p] / self.dx
            base = int(Xp - 0.5)
            fx = Xp - base
            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
            new_v = ti.Vector.zero(float, 2)
            new_C = ti.Matrix.zero(float, 2, 2)
            for i, j in ti.static(ti.ndrange(3, 3)):
                offset = ti.Vector([i, j])
                dpos = (offset - fx) * self.dx
                weight = w[i].x * w[j].y
                g_v = self.grid_v[base + offset]
                new_v += weight * g_v
                new_C += 4 * weight * g_v.outer_product(dpos) / self.dx**2
            self.v[p] = new_v
            self.x[p] += self.dt * self.v[p]
            self.J[p] *= 1 + self.dt * new_C.trace()
            self.C[p] = new_C

    @ti.kernel
    def init(self):
        # set initial pos based on n_particles
        width = 0.05 + 0.35 * (self.n_particles / 1000)
        height = 0.2 + 0.5 * (self.n_particles / 1000)
        x_offset = ti.random() * (1 - width)
        y_offset = ti.random() * (1 - height)
        for i in range(self.n_particles):
            self.x[i] = [
                ti.random() * width + x_offset,
                ti.random() * height + y_offset,
            ]
            self.v[i] = [0, -1]
            self.J[i] = 1


def main():
    gui = ti.GUI("MPM88")
    count = 0
    while gui.running and not gui.get_event(gui.ESCAPE):
        if count % 900 == 0:
            n = random.randint(200, 1200)
            mpm = MPM88(n)
            print("Num particles:", n)
        for s in range(50):
            mpm.substep()
        gui.clear(0x112F41)
        gui.circles(mpm.x.to_numpy(), radius=1.5, color=0x068587)
        gui.show()
        count += 1


if __name__ == "__main__":
    ti.init(arch=ti.gpu)
    main()