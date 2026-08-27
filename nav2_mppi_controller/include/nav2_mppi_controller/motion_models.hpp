// Copyright (c) 2022 Samsung Research America, @artofnothingness Alexey Budyakov
// Copyright (c) 2025 Open Navigation LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef NAV2_MPPI_CONTROLLER__MOTION_MODELS_HPP_
#define NAV2_MPPI_CONTROLLER__MOTION_MODELS_HPP_

#include <Eigen/Dense>

#include <cmath>
#include <cstdint>
#include <string>
#include <algorithm>
#include <vector>

#include "nav2_mppi_controller/models/control_sequence.hpp"
#include "nav2_mppi_controller/models/state.hpp"
#include "nav2_mppi_controller/models/constraints.hpp"

#include "nav2_mppi_controller/tools/parameters_handler.hpp"
#include "nav2_mppi_controller/tools/velocity_limits.hpp"

namespace mppi
{

/**
 * @class mppi::MotionModel
 * @brief Abstract pluginlib class for modeling a vehicle
 */
class MotionModel
{
public:
  /**
    * @brief Constructor for mppi::MotionModel
    */
  MotionModel() = default;

  /**
    * @brief Destructor for mppi::MotionModel
    */
  virtual ~MotionModel() = default;

  /**
   * @brief Initialize motion model on bringup.
   * @param param_handler Pointer to the shared parameters handler
   * @param plugin_name   Namespaced name of this plugin instance
   */
  virtual void initialize(
    ParametersHandler * /*param_handler*/,
    const std::string & /*plugin_name*/)
  {}

  /**
    * @brief Initialize motion model on bringup and set required variables
    * @param control_constraints Constraints on control
    * @param model_dt duration of a time step
    */
  void setConstraints(
    const models::ControlConstraints & control_constraints, float model_dt,
    float model_delay_vx, float model_delay_vy, float model_delay_wz, bool clamp_raw_controls)
  {
    control_constraints_ = control_constraints;
    model_dt_ = model_dt;
    model_delay_vx_ = model_delay_vx;
    model_delay_vy_ = model_delay_vy;
    model_delay_wz_ = model_delay_wz;
    clamp_raw_controls_ = clamp_raw_controls;

    cmd_history_vx_.resize(offsetSteps(model_delay_vx_), 0.0f);
    cmd_history_vy_.resize(offsetSteps(model_delay_vy_), 0.0f);
    cmd_history_wz_.resize(offsetSteps(model_delay_wz_), 0.0f);
  }

  /**
    * @brief Push the most recently published command to the per-axis history
    *        ring buffers. Called once per controller cycle from the optimizer.
    */
  void pushCommandHistory(float vx, float vy, float wz)
  {
    pushOne(cmd_history_vx_, vx);
    pushOne(cmd_history_vy_, vy);
    pushOne(cmd_history_wz_, wz);
  }

  /**
    * @brief Zero the ring buffers
    */
  void clearCommandHistory()
  {
    std::fill(cmd_history_vx_.begin(), cmd_history_vx_.end(), 0.0f);
    std::fill(cmd_history_vy_.begin(), cmd_history_vy_.end(), 0.0f);
    std::fill(cmd_history_wz_.begin(), cmd_history_wz_.end(), 0.0f);
  }

  /**
   * @brief With input velocities, find the vehicle's output velocities
   * @param state Contains control velocities to use to populate vehicle velocities
   */
  virtual void predict(models::State & state)
  {
    const bool is_holo = isHolonomic();
    models::Control min_deltas, max_deltas;
    accelDeltas(model_dt_, min_deltas, max_deltas);
    unsigned int n_cols = state.vx.cols();

    // Set dynamic limits to the platform velocities from the raw controls sampling
    for (unsigned int i = 1; i < n_cols; i++) {
      auto lower_bound_vx = (state.vx.col(i - 1) >
        0).select(
        state.vx.col(i - 1) + min_deltas.vx,
        state.vx.col(i - 1) - max_deltas.vx);
      auto upper_bound_vx = (state.vx.col(i - 1) >
        0).select(
        state.vx.col(i - 1) + max_deltas.vx,
        state.vx.col(i - 1) - min_deltas.vx);
      state.vx.col(i) = state.cvx.col(i - 1)
        .cwiseMax(lower_bound_vx)
        .cwiseMin(upper_bound_vx);
      if (clamp_raw_controls_) {
        state.cvx.col(i - 1) = state.vx.col(i);
      }

      state.wz.col(i) = state.cwz.col(i - 1)
        .cwiseMax(state.wz.col(i - 1) - max_deltas.wz)
        .cwiseMin(state.wz.col(i - 1) + max_deltas.wz);
      if (clamp_raw_controls_) {
        state.cwz.col(i - 1) = state.wz.col(i);
      }

      if (is_holo) {
        auto lower_bound_vy = (state.vy.col(i - 1) >
          0).select(
          state.vy.col(i - 1) + min_deltas.vy,
          state.vy.col(i - 1) - max_deltas.vy);
        auto upper_bound_vy = (state.vy.col(i - 1) >
          0).select(
          state.vy.col(i - 1) + max_deltas.vy,
          state.vy.col(i - 1) - min_deltas.vy);
        state.vy.col(i) = state.cvy.col(i - 1)
          .cwiseMax(lower_bound_vy)
          .cwiseMin(upper_bound_vy);
        if (clamp_raw_controls_) {
          state.cvy.col(i - 1) = state.vy.col(i);
        }
      }
    }

    const unsigned int offset_vx = std::floor((model_delay_vx_ / model_dt_) + 0.5);
    const unsigned int offset_vy = std::floor((model_delay_vy_ / model_dt_) + 0.5);
    const unsigned int offset_wz = std::floor((model_delay_wz_ / model_dt_) + 0.5);

    if (offset_vx > 0u || offset_wz > 0u || (is_holo && offset_vy > 0u)) {
      applyDelayShift(state, is_holo, offset_vx, offset_vy, offset_wz);
    }
  }

  /**
   * @brief Whether the motion model is holonomic, using Y axis
   * @return Bool If holonomic
   */
  virtual bool isHolonomic() const = 0;

  /**
   * @brief Constrain a control sequence to the velocities the vehicle can actually reach
   * @param control_sequence Control sequence to constrain in place
   * @param initial_speed Measured speed the sequence has to ramp away from
   * @param first_dt Duration of the first time step, the controller period
   * @param shift_control_sequence Whether index 0 is "now" and is not sent to the robot
   */
  void constrainControlSequence(
    models::ControlSequence & control_sequence,
    const geometry_msgs::msg::Twist & initial_speed,
    const float first_dt, const bool shift_control_sequence)
  {
    const bool is_holo = isHolonomic();

    // Start from the current speed to create inter-iteration dynamic feasibility
    models::Control last{
      static_cast<float>(initial_speed.linear.x),
      is_holo ? static_cast<float>(initial_speed.linear.y) : 0.0f,
      static_cast<float>(initial_speed.angular.z)};

    // When shifting, vx(0) is "now" and not sent. Pin it so vx(1), the sent command,
    // is exactly one constraint step from current speed
    if (shift_control_sequence) {
      control_sequence.vx(0) = last.vx;
      control_sequence.wz(0) = last.wz;
      if (is_holo) {
        control_sequence.vy(0) = last.vy;
      }
    }

    // Use the controller period for t=0 to realistically model physical limits, then switch to
    // the MPC model_dt for intra-iteration feasibility
    models::Control min_deltas, max_deltas;
    accelDeltas(first_dt, min_deltas, max_deltas);

    for (unsigned int i = 0; i != control_sequence.vx.size(); i++) {
      if (i == 1) {
        accelDeltas(model_dt_, min_deltas, max_deltas);
      }

      models::Control curr{control_sequence.vx(i),
        is_holo ? control_sequence.vy(i) : 0.0f, control_sequence.wz(i)};
      constrainVelocityStep(curr, last, min_deltas, max_deltas);

      control_sequence.vx(i) = curr.vx;
      control_sequence.wz(i) = curr.wz;
      if (is_holo) {
        control_sequence.vy(i) = curr.vy;
      }

      last = curr;
    }
  }

protected:
  /**
   * @brief Constrain one time step of a control sequence to the velocities reachable from the
   *        previous one, bounding each axis by its own velocity and acceleration limits
   * @param curr Requested velocities, constrained in place
   * @param last Velocities at the previous time step, assumed to be reachable
   * @param min_deltas Most negative change in speed the acceleration limits allow, per axis
   * @param max_deltas Most positive change in speed the acceleration limits allow, per axis
   */
  virtual void constrainVelocityStep(
    models::Control & curr, const models::Control & last,
    const models::Control & min_deltas, const models::Control & max_deltas) const
  {
    curr.vx = utils::clamp(control_constraints_.vx_min, control_constraints_.vx_max, curr.vx);
    curr.vx = utils::clampVelocityByAccel(last.vx, curr.vx, min_deltas.vx, max_deltas.vx);

    curr.vy = utils::clamp(-control_constraints_.vy, control_constraints_.vy, curr.vy);
    curr.vy = utils::clampVelocityByAccel(last.vy, curr.vy, min_deltas.vy, max_deltas.vy);

    curr.wz = utils::clamp(-control_constraints_.wz, control_constraints_.wz, curr.wz);
    curr.wz = utils::clampVelocityByAccel(last.wz, curr.wz, min_deltas.wz, max_deltas.wz);
  }

  /**
   * @brief Change in speed the acceleration limits allow over a time step
   * @param dt Duration of the time step
   * @param min_deltas Most negative change in speed, per axis
   * @param max_deltas Most positive change in speed, per axis
   */
  void accelDeltas(
    const float dt, models::Control & min_deltas, models::Control & max_deltas) const
  {
    min_deltas = {dt * control_constraints_.ax_min, dt * control_constraints_.ay_min,
      -dt * control_constraints_.az_max};
    max_deltas = {dt * control_constraints_.ax_max, dt * control_constraints_.ay_max,
      dt * control_constraints_.az_max};
  }

  /**
    * @brief Apply the per-axis input-delay shift to velocity rollout.
    *
    * For j in [1, offset) — the delay window — fill `dst` with history[j]
    */
  void applyDelayShift(
    models::State & state, bool is_holo,
    unsigned int offset_vx, unsigned int offset_vy, unsigned int offset_wz) const
  {
    auto shift = [](Eigen::ArrayXXf & velocities, unsigned int offset,
      const std::vector<float> & history) {
        const unsigned int cols = static_cast<unsigned int>(velocities.cols());
        if (offset == 0u || cols == 0u) {
          return;
        }

        // Shift cols in-place right-to-left by offset
        for (unsigned int k = (offset < cols) ? cols - offset : 0u; k > 0; --k) {
          velocities.col(offset + k - 1) = velocities.col(k);
        }

        // Fill delay window cols with the in-flight commands from history.
        const unsigned int end = std::min(offset, cols);
        for (unsigned int j = 1; j < end; ++j) {
          velocities.col(j).setConstant(history[j]);
        }
      };

    shift(state.vx, offset_vx, cmd_history_vx_);
    shift(state.wz, offset_wz, cmd_history_wz_);

    if (is_holo) {
      shift(state.vy, offset_vy, cmd_history_vy_);
    }
  }

  /**
    * @brief Convert a delay in seconds to an offset in number of rollout steps, rounding to the nearest step.
    */
  std::size_t offsetSteps(float delay) const
  {
    if (delay <= 0.0f || model_dt_ <= 0.0f) {
      return 0u;
    }
    return static_cast<std::size_t>(std::floor(delay / model_dt_ + 0.5f));
  }

  /**
    * @brief Push a value to the back of a ring buffer and rotate the elements.
    */
  static void pushOne(std::vector<float> & buf, float v)
  {
    if (buf.empty()) {return;}
    std::rotate(buf.begin(), buf.begin() + 1, buf.end());
    buf.back() = v;
  }

  float model_dt_{0.0};
  float model_delay_vx_{0.0};
  float model_delay_vy_{0.0};
  float model_delay_wz_{0.0};
  bool clamp_raw_controls_{false};

  // Per-axis ring buffer of recently published commands
  std::vector<float> cmd_history_vx_;
  std::vector<float> cmd_history_vy_;
  std::vector<float> cmd_history_wz_;

  models::ControlConstraints control_constraints_{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 0.0f};
};

/**
 * @class mppi::AckermannMotionModel
 * @brief Ackermann motion model
 */
class AckermannMotionModel : public MotionModel
{
public:
  /**
    * @brief Constructor for mppi::AckermannMotionModel
    */
  AckermannMotionModel() = default;

  /**
   * @brief Initialize motion model.
   * @param param_handler Pointer to the shared parameters handler
   * @param plugin_name   Namespaced name of this plugin instance
   */
  void initialize(
    ParametersHandler * param_handler,
    const std::string & plugin_name) override
  {
    auto getParam = param_handler->getParamGetter(plugin_name);
    getParam(min_turning_r_, "min_turning_r", 0.2f);
  }

  /**
   * @brief Whether the motion model is holonomic, using Y axis
   * @return Bool If holonomic
   */
  bool isHolonomic() const override
  {
    return false;
  }

  /**
   * @brief Get minimum turning radius of ackermann drive
   * @return Minimum turning radius
   */
  float getMinTurningRadius() const {return min_turning_r_;}

protected:
  /**
   * @brief Constrain one time step to the reachable velocities, which for ackermann are bounded
   *        by the per-axis limits and by the minimum turning radius, r * |wz| <= |vx|
   */
  void constrainVelocityStep(
    models::Control & curr, const models::Control & last,
    const models::Control & min_deltas, const models::Control & max_deltas) const override
  {
    // Aim at the requested velocity brought inside the turning radius envelope. The radius is
    // floored because it is an unvalidated parameter, and a radius of zero at a standstill would
    // otherwise divide zero by zero and poison the rest of the horizon with NaN.
    const float vx_target = utils::clamp(
      control_constraints_.vx_min, control_constraints_.vx_max, curr.vx);
    const float wz_bound = std::min(
      std::fabs(vx_target) / std::max(min_turning_r_, 1e-6f), control_constraints_.wz);
    const float wz_target = utils::clamp(-wz_bound, wz_bound, curr.wz);

    // Then step toward it. Unlike a box or an ellipse the turning radius envelope is not convex
    // across vx = 0, so the step is bounded by the envelope as well as by the acceleration
    // limits, rather than relying on the target being reachable.
    const float dvx = vx_target - last.vx;
    const float dwz = wz_target - last.wz;
    const float alpha = std::min(
      {1.0f,
        utils::reachableFraction(last.vx, vx_target, min_deltas.vx, max_deltas.vx),
        utils::reachableFraction(last.wz, wz_target, min_deltas.wz, max_deltas.wz),
        turningRadiusFraction(last.vx, last.wz, dvx, dwz)});

    curr.vx = last.vx + alpha * dvx;
    curr.wz = last.wz + alpha * dwz;
  }

private:
  /**
   * @brief Largest fraction of a step away from (vx_last, wz_last) that keeps the minimum
   *        turning radius satisfied
   * @param vx_last Longitudinal velocity at the previous time step
   * @param wz_last Angular velocity at the previous time step
   * @param dvx Requested change in longitudinal velocity
   * @param dwz Requested change in angular velocity
   * @return Fraction in [0, 1]
   */
  float turningRadiusFraction(
    const float vx_last, const float wz_last, const float dvx, const float dwz) const
  {
    // On the side of vx = 0 that the step starts from, |vx| is linear in the step fraction a, so
    // r * |wz| <= |vx| is a pair of linear bounds on a,
    //     a * ( r*dwz - s*dvx) <= s*vx_last - r*wz_last
    //     a * (-r*dwz - s*dvx) <= s*vx_last + r*wz_last
    // with s the sign of that side. Only a positive coefficient bounds a from above, and both
    // right hand sides are non-negative because a = 0 is always feasible. The bounds hold at or
    // before vx reaches 0, which is correct: an ackermann vehicle at a standstill cannot turn.
    const float s = vx_last != 0.0f ? std::copysign(1.0f, vx_last) :
      (dvx != 0.0f ? std::copysign(1.0f, dvx) : 1.0f);
    const float r = min_turning_r_;

    float fraction = 1.0f;
    const auto boundBy = [&fraction](const float coeff, const float rhs) {
        if (coeff > 0.0f) {
          fraction = std::min(fraction, std::max(rhs, 0.0f) / coeff);
        }
      };
    boundBy(r * dwz - s * dvx, s * vx_last - r * wz_last);
    boundBy(-r * dwz - s * dvx, s * vx_last + r * wz_last);

    return fraction;
  }

  float min_turning_r_{0.0f};
};

/**
 * @class mppi::DiffDriveMotionModel
 * @brief Differential drive motion model
 */
class DiffDriveMotionModel : public MotionModel
{
public:
  /**
    * @brief Constructor for mppi::DiffDriveMotionModel
    */
  DiffDriveMotionModel() = default;

  /**
   * @brief Whether the motion model is holonomic, using Y axis
   * @return Bool If holonomic
   */
  bool isHolonomic() const override
  {
    return false;
  }
};

/**
 * @class mppi::OmniMotionModel
 * @brief Omnidirectional motion model
 */
class OmniMotionModel : public MotionModel
{
public:
  /**
    * @brief Constructor for mppi::OmniMotionModel
    */
  OmniMotionModel() = default;

  /**
   * @brief Initialize motion model.
   * @param param_handler Pointer to the shared parameters handler
   * @param plugin_name   Namespaced name of this plugin instance
   */
  void initialize(
    ParametersHandler * param_handler,
    const std::string & plugin_name) override
  {
    auto getParam = param_handler->getParamGetter(plugin_name);
    getParam(use_velocity_ellipse_scaling_, "use_velocity_ellipse_scaling", true);
  }

  /**
   * @brief Whether the motion model is holonomic, using Y axis
   * @return Bool If holonomic
   */
  bool isHolonomic() const override
  {
    return true;
  }

  /**
   * @brief Whether the combined translational velocity is bounded by the ellipse spanned by the
   *        per-axis limits, rather than each axis being bounded on its own
   *
   * Bounding the combination means a diagonal command cannot travel faster than a straight one,
   * which incentivizes forward over diagonal travel when the optimizer asks for full velocity.
   * Bounding each axis on its own lets the combined speed reach sqrt(vx_max² + vy_max²).
   *
   * @return Bool If the combined translational velocity is bounded
   */
  bool useVelocityEllipseScaling() const
  {
    return use_velocity_ellipse_scaling_;
  }

  /**
   * @brief Amount by which velocities exceed the velocity ellipse, along their direction of
   *        travel, for use as a constraint violation cost
   * @param vx Longitudinal velocities
   * @param vy Lateral velocities
   * @return Excess speed in m/s, zero where feasible
   */
  template<typename Derived>
  auto getTranslationalVelocityViolation(
    const Eigen::ArrayBase<Derived> & vx, const Eigen::ArrayBase<Derived> & vy) const
  {
    // (vx/vx_max)² + (vy/vy_max)², which is 1 exactly on the ellipse. The forward and reverse
    // semi-axes differ, so vx is split by direction of travel: one of the two terms is zero.
    // Naming subexpressions would not save any work here, since an Eigen expression used twice
    // is evaluated twice.
    const auto normalized_sq =
      invSquare(control_constraints_.vx_max) * vx.max(0.0f).square() +
      invSquare(control_constraints_.vx_min) * vx.min(0.0f).square() +
      invSquare(control_constraints_.vy) * vy.square();

    // |v| - |v| / sqrt(normalized_sq), clamped so that feasible velocities cost nothing
    return (vx.square() + vy.square()).sqrt() * (1.0f - normalized_sq.max(1.0f).rsqrt());
  }

protected:
  /**
   * @brief Constrain one time step to the reachable velocities, which for omni are bounded by
   *        the angular limits and by the translational velocity ellipse
   */
  void constrainVelocityStep(
    models::Control & curr, const models::Control & last,
    const models::Control & min_deltas, const models::Control & max_deltas) const override
  {
    // The ellipse only bounds translation, so wz is bounded on its own
    curr.wz = utils::clamp(-control_constraints_.wz, control_constraints_.wz, curr.wz);
    curr.wz = utils::clampVelocityByAccel(last.wz, curr.wz, min_deltas.wz, max_deltas.wz);

    if (!use_velocity_ellipse_scaling_) {
      curr.vx = utils::clamp(control_constraints_.vx_min, control_constraints_.vx_max, curr.vx);
      curr.vx = utils::clampVelocityByAccel(last.vx, curr.vx, min_deltas.vx, max_deltas.vx);
      curr.vy = utils::clamp(-control_constraints_.vy, control_constraints_.vy, curr.vy);
      curr.vy = utils::clampVelocityByAccel(last.vy, curr.vy, min_deltas.vy, max_deltas.vy);
      return;
    }

    // An axis limited to zero collapses the ellipse onto the remaining axes, which a radial
    // projection cannot reach. Drop such an axis first so the others can still be commanded.
    if (control_constraints_.vy <= 1e-6f) {
      curr.vy = 0.0f;
    }
    if (control_constraints_.vx_max <= 1e-6f) {
      curr.vx = std::min(curr.vx, 0.0f);
    }
    if (control_constraints_.vx_min >= -1e-6f) {
      curr.vx = std::max(curr.vx, 0.0f);
    }

    // Aim at the requested velocity projected radially onto the ellipse, which preserves the
    // commanded direction of travel and implies the per-axis limits, then step toward that
    // target as far as the acceleration limits allow. The ellipse is convex, so a step from a
    // reachable velocity toward a target inside it cannot leave it: both limits hold at once,
    // without a second projection that would discard the ramp.
    const float scale = velocityEllipseScale(curr.vx, curr.vy);
    const float dvx = curr.vx * scale - last.vx;
    const float dvy = curr.vy * scale - last.vy;

    const float alpha = std::min(
      {1.0f,
        utils::reachableFraction(last.vx, last.vx + dvx, min_deltas.vx, max_deltas.vx),
        utils::reachableFraction(last.vy, last.vy + dvy, min_deltas.vy, max_deltas.vy)});

    curr.vx = last.vx + alpha * dvx;
    curr.vy = last.vy + alpha * dvy;
  }

  /**
   * @brief Scale that projects a velocity onto the velocity ellipse, leaving feasible ones alone
   * @param vx Longitudinal velocity
   * @param vy Lateral velocity
   * @return Scale factor in (0, 1]
   */
  float velocityEllipseScale(const float vx, const float vy) const
  {
    const float vx_limit = vx >= 0.0f ?
      control_constraints_.vx_max : control_constraints_.vx_min;
    const float normalized_sq =
      invSquare(vx_limit) * vx * vx + invSquare(control_constraints_.vy) * vy * vy;

    return 1.0f / std::sqrt(std::max(normalized_sq, 1.0f));
  }

  /**
   * @brief Inverse square of a velocity limit, floored so that a zero limit cannot divide by zero
   * @param limit Velocity limit of either sign
   * @return 1 / limit², at most 1e12
   */
  static float invSquare(const float limit)
  {
    constexpr float min_limit = 1e-6f;
    return 1.0f / std::max(limit * limit, min_limit * min_limit);
  }

  bool use_velocity_ellipse_scaling_{true};
};

}  // namespace mppi

#endif  // NAV2_MPPI_CONTROLLER__MOTION_MODELS_HPP_
