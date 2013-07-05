from dolfin import *
from numpy import exp
import numpy
#parameters["linear_algebra_backend"] = "Epetra"
from time import time as clock

parameters.form_compiler.optimize = True
parameters.form_compiler.cpp_optimize = True

from fiberrules import *

set_log_level(15)

from dolfin import *
import numpy

code = """
#include <vector>
#include <cmath>
namespace dolfin
{

class PurkinjeSubdomain: public SubDomain
{
public:

  PurkinjeSubdomain(const std::vector<dolfin::Point>& points, double radius) :
   only_on_boundary(false), _points(points), _radius_sqr(std::pow(radius, 2))
  {}

  /// Return true for points inside the sub domain
  bool inside(const Array<double>& x, bool on_boundary) const
  {
    if (only_on_boundary && !on_boundary)
      return false;

    return within_point(x)>=0;
  }

  int within_point(const Array<double>& x) const
  {
      
    double dist_sqr=1.0/DOLFIN_EPS;
    int point_i = -1;
    for (unsigned int i=0; i<_points.size(); i++)
    {
      double dist_sqr_i = 0.0;
      for (unsigned int j=0; j < x.size(); j++)
        dist_sqr_i += std::pow(_points[i].coordinates()[j]-x[j], 2);

      if (dist_sqr_i < dist_sqr)
      {
        point_i = i;
        dist_sqr = dist_sqr_i;
      }
    }

    return dist_sqr < _radius_sqr ? point_i : -1;

  }

  bool only_on_boundary;

private:
  
  std::vector<dolfin::Point> _points;
  double _radius_sqr;
};

class ActivationTimes: public Expression
{
public:

  ActivationTimes(PurkinjeSubdomain& subdomain,
                  std::vector<double> values) :
   _subdomain(subdomain), _activation_times(values)
  {
  }

  /// Return true for points inside the sub domain
  void eval(Array<double>& values, const Array<double>& x) const
  {
    const int point_i = _subdomain.within_point(x);
    if (point_i>=0)
    {
      dolfin_assert(point_i<_activation_times.size());
      values[0] = _activation_times[point_i];
    }
    else
      values[0] = 1/DOLFIN_EPS;
  }

private:

  PurkinjeSubdomain _subdomain;
  std::vector<double> _activation_times;
  
};

}
"""


def compute_activation(casename = "", timing = None, plot_solution = False):

    t0 = clock();

    parameters.num_threads = 8
    print casename
    mesh = Mesh(casename + "heart.xml.gz")

    W = VectorFunctionSpace(mesh, "CG", 1)
    
    fibers = Function(W)
    normals = Function(W)
    cross = Function(W)

    file_f = File(casename + "fibers.xml")
    file_n = File(casename + "normals.xml")
    file_c = File(casename + "cross.xml")
   
    file_f >> fibers
    file_n >> normals
    file_c >> cross
    
    def outer(v):
	return as_matrix(( (v[0]*v[0], v[0]*v[1], v[0]*v[2]), (v[1]*v[0], v[1]*v[1], v[1]*v[2]), (v[2]*v[0], v[2]*v[1], v[2]*v[2])));

    sigma_l = 0.25
    sigma_t = 0.04/10
    sigma_n = 0.04/10

    M = sigma_l*outer(fibers) + sigma_t*outer(cross) + sigma_n*outer(normals)
    #M = 10*Constant(((0.025,0,0),(0,0.004,0),(0,0,0.004)))

    sites = numpy.loadtxt(casename +"purkinje_transformed.txt")
    if len(sites.shape)==1:
	sites = [sites]
    if timing == None:
	# use all, and start all at t=0
	time = numpy.zeros(sites.shape[0])
    else:
	cut_off = 60; # Drop late points, that the optimizer might suggest
	idx = (timing>=0)*(timing<cut_off)
	sites = sites[idx]
	time = timing[idx]
	#sites = sites[numpy.nonzero(timing>=0)]
	#time = timing[numpy.nonzero(timing>=0)]

    radius = 0.5


    V = FunctionSpace(mesh, 'CG', 1)
    v = TestFunction(V)
    u = TrialFunction(V)


    tau = Constant(2)
    c_0 = Constant(2.5)
    t1 = clock()

    points = [];
    for p in sites:
	points += [Point(p[0],p[1],p[2])]
    radius = 1.0
    ext_module = compile_extension_module(code)
    subdomain = ext_module.PurkinjeSubdomain(points, radius)
    activation_times = ext_module.ActivationTimes(subdomain, time)
    #activation_times = Constant(0)
    bc = DirichletBC(V, activation_times, subdomain)    
	
    t2 = clock();


    T = Function(V)

    # Initialization problem to get good initial guess for nonlinear problem:
    F1 = inner(grad(u),  M*grad(v))*dx - tau*tau*v*dx

    print "Trying to solve for initial guess...\n"
    solve(lhs(F1)==rhs(F1), T, bc)
    print "Done making guess\n"
    
    t3 = clock();

    FDM = (c_0*(sqrt(inner(grad(T), M*grad(T))) - tau)*v + inner(grad(T), M*grad(v)))*dx

    try:
	solve(FDM==0, T, bc, solver_parameters={"linear_solver": "lu", "newton_solver": {"relaxation_parameter":0.75, "relative_tolerance":1e-5, "maximum_iterations":30}})
    except RuntimeError:
	print "Newton solver did not converge."


    print "Max distance:", T.vector().max()
    print "Min distance:", T.vector().min()
    T.vector().array().tofile(casename + "activation_times.txt");
    
    t4 = clock();
    print "solve", t1-t0,t2-t1,t3-t2, t4-t3

    #f = File("distance.pvd")
    #f << T
    if plot_solution:
	plot(T)
	interactive()

if __name__ == '__main__':

    compute_activation()
