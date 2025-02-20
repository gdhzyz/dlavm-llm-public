from ..adr import Functor, Call


class EliminateReshape(Functor):

    class CheckEliminate(Functor):
        # analysis reshape operator as (x, y) to (x, 1, y)
        def reshape_squeeze(self, shape, new_shape):
            axes, sidx = -1, 0
            lshape = shape if len(shape) > len(new_shape) else new_shape
            sshape = new_shape if len(shape) > len(new_shape) else shape
            for n in range(len(lshape)):
                if lshape[n] == 1 and axes == -1:
                    axes = n
                elif lshape[n] == sshape[sidx]:
                    sidx += 1
                else:
                    return False, axes
            if axes != -1:
                return True, axes
            return False, axes

        def visit_vm(self, expr):
            if expr.op.name == "accel.reshape":
                state, axes = self.reshape_squeeze(expr.args[0].checked_type.shape, expr.checked_type.shape)
                if state:
                    if hasattr(self, "out_axes"):
                        if self.out_axes == axes:
                            if hasattr(self, "last_expr"):
                                self.last_expr.args[0] = expr.args[0]
                                return self.last_expr
                        return self.org_expr
                    else:
                        self.org_expr = expr
                        self.out_axes = axes
                        return self.visit(expr.args[0])
            if hasattr(self, "org_expr"):
                return self.org_expr
            return expr

        def visit_tupleitem(self, expr):
            new_expr = self.visit(expr.expr)
            return new_expr

        def visit_call(self, expr):
            if expr.op.name in ["accel.hbm.mvm", "accel.hbm.mvm_bn", "accel.hbm.mvm_bn_res"] and hasattr(self, "out_axes"):
                self.last_expr = expr
                return self.visit(expr.args[0])
            else:
                return self.org_expr

    def visit_vm(self, expr):
        new_args = [self.visit(arg) for arg in expr.args]
        expr.args = new_args
        new_expr = self.CheckEliminate().visit(expr)
        return new_expr


def eliminate_reshape(expr):
    return EliminateReshape().visit(expr)
