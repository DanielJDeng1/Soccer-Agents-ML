using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;

[RequireComponent(typeof(Rigidbody))]
public class SoccerAgent : Agent
{
    [Header("Team")]
    public Team team = Team.Blue;
    public SoccerEnvController envController;

    [Header("Environment References")]
    public Transform ball;
    public Rigidbody ballRb;
    public Transform opponentGoal;
    public Transform ownGoal;

    [Header("Team and Opponent Positions")]
    public List<Transform> teammates = new List<Transform>();
    public List<Transform> opponents = new List<Transform>();

    [Header("Movement Settings")]
    public float moveSpeed = 7f;
    public float acceleration = 18f;
    public float rotationSmoothing = 15f;
    public float braking = 8f;
    public float angularDamping = 8f;

    [Header("Reward Parameters")]
    [Tooltip("Base step cost. Sparse goal rewards are handled by SoccerEnvController.")]
    public float existencePenalty = -0.0005f;

    [Header("Engagement Shaping")]
    [Tooltip("Reward for closest agent approaching ball. Prevents multi-agent clustering.")]
    public float ballApproachShapingScale = 0.05f;
    public float maxBallApproachRewardPerStep = 0.003f;

    [Tooltip("Distance threshold to stop approach rewards and transition to engagement.")]
    public float ballApproachShapingCutoff = 1.0f;

    [Header("Anti Crowding")]
    [Tooltip("Penalty radius around ball to prevent teammates from bunching up.")]
    public float teammateCrowdingRadius = 2.0f;
    public float teammateCrowdingPenalty = -0.01f;

    [Header("Observation Normalization")]
    public float fieldNormalizationDistance = 20f;
    public float maxBallSpeedForNormalization = 15f;

    private Rigidbody agentRb;
    private float previousDistanceToBall;
    private bool hasPreviousDistanceToBall;
    private bool referencesValid;

    public override void Initialize()
    {
        agentRb = GetComponent<Rigidbody>();
        agentRb.angularDamping = angularDamping;

        var missing = new List<string>();
        if (ball == null) missing.Add("ball");
        if (ballRb == null) missing.Add("ballRb");
        if (opponentGoal == null) missing.Add("opponentGoal");
        if (ownGoal == null) missing.Add("ownGoal");

        if (missing.Count > 0)
        {
            Debug.LogError($"[{name}] SoccerAgent missing required references: {string.Join(", ", missing)}. " +
                            "Check the Inspector — script reloads can reset serialized fields.");
            referencesValid = false;
        }
        else
        {
            referencesValid = true;
        }
    }

    public override void OnEpisodeBegin()
    {
        agentRb.linearVelocity = Vector3.zero;
        agentRb.angularVelocity = Vector3.zero;
        hasPreviousDistanceToBall = false;
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        if (!referencesValid) return;

        // ball state (6 floats)
        sensor.AddObservation((ball.position - transform.position) / fieldNormalizationDistance);
        sensor.AddObservation(ballRb.linearVelocity / maxBallSpeedForNormalization);

        // relative goal offsets (6 floats)
        sensor.AddObservation((opponentGoal.position - transform.position) / fieldNormalizationDistance);
        sensor.AddObservation((ownGoal.position - transform.position) / fieldNormalizationDistance);

        // Kinematics and orientation (6 floats)
        sensor.AddObservation(agentRb.linearVelocity / moveSpeed);
        sensor.AddObservation(transform.forward);

        // teammates pad with zero vectors if missing
        foreach (var teammate in teammates)
        {
            sensor.AddObservation(teammate != null
                ? (teammate.position - transform.position) / fieldNormalizationDistance
                : Vector3.zero);
        }

        // opponents, 3 floats each
        foreach (var opponent in opponents)
        {
            sensor.AddObservation(opponent != null
                ? (opponent.position - transform.position) / fieldNormalizationDistance
                : Vector3.zero);
        }

        // normalized pitch coordinates of 2 float vals
        if (envController != null && envController.pitchBoundsCollider != null)
        {
            Bounds bounds = envController.pitchBoundsCollider.bounds;
            float normX = Mathf.InverseLerp(bounds.min.x, bounds.max.x, ball.position.x) * 2f - 1f;
            float normZ = Mathf.InverseLerp(bounds.min.z, bounds.max.z, ball.position.z) * 2f - 1f;
            sensor.AddObservation(normX);
            sensor.AddObservation(normZ);
        }
        else
        {
            sensor.AddObservation(0f);
            sensor.AddObservation(0f);
        }
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        if (!referencesValid) return;

        AddReward(existencePenalty);
        ApplyBallApproachShaping();
        ApplyTeammateCrowdingPenalty();

        float moveX = Mathf.Clamp(actions.ContinuousActions[0], -1f, 1f);
        float moveZ = Mathf.Clamp(actions.ContinuousActions[1], -1f, 1f);

        Vector3 inputVector = new Vector3(moveX, 0f, moveZ);
        Vector3 horizontalVelocity = new Vector3(agentRb.linearVelocity.x, 0f, agentRb.linearVelocity.z);

        if (inputVector.sqrMagnitude > 0.01f)
        {
            Vector3 worldMoveDir = inputVector.normalized;
            Vector3 desiredVelocity = worldMoveDir * moveSpeed;
            Vector3 velocityError = desiredVelocity - horizontalVelocity;

            agentRb.AddForce(velocityError * acceleration, ForceMode.Acceleration);

            Quaternion targetRotation = Quaternion.LookRotation(worldMoveDir, Vector3.up);
            Quaternion smoothedRotation = Quaternion.Slerp(
                agentRb.rotation,
                targetRotation,
                Time.fixedDeltaTime * rotationSmoothing
            );

            agentRb.MoveRotation(smoothedRotation);
        }
        else
        {
            agentRb.AddForce(-horizontalVelocity * braking, ForceMode.Acceleration);
        }
    }

    // Reward active ball pursuit
    private void ApplyBallApproachShaping()
    {
        if (ballApproachShapingScale <= 0f) return;

        if (!IsClosestTeammateToBall())
        {
            hasPreviousDistanceToBall = false;
            return;
        }

        float currentDistance = Vector3.Distance(transform.position, ball.position);
        if (currentDistance <= ballApproachShapingCutoff)
        {
            hasPreviousDistanceToBall = false;
            return;
        }

        if (hasPreviousDistanceToBall)
        {
            float delta = previousDistanceToBall - currentDistance;
            float reward = Mathf.Clamp(delta * ballApproachShapingScale, -maxBallApproachRewardPerStep, maxBallApproachRewardPerStep);
            AddReward(reward);
        }

        previousDistanceToBall = currentDistance;
        hasPreviousDistanceToBall = true;
    }

    // penalize secondary agents crowding the ball
    private void ApplyTeammateCrowdingPenalty()
    {
        if (teammateCrowdingPenalty >= 0f) return;

        float myDistance = Vector3.Distance(transform.position, ball.position);
        if (myDistance > teammateCrowdingRadius) return;

        foreach (var teammate in teammates)
        {
            if (teammate == null) continue;
            if (Vector3.Distance(teammate.position, ball.position) <= teammateCrowdingRadius)
            {
                AddReward(teammateCrowdingPenalty);
                return; // early exit once crowding condition is met
            }
        }
    }

    private bool IsClosestTeammateToBall()
    {
        float myDistance = Vector3.Distance(transform.position, ball.position);
        foreach (var teammate in teammates)
        {
            if (teammate == null) continue;
            if (Vector3.Distance(teammate.position, ball.position) < myDistance) return false;
        }
        return true;
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var continuousActions = actionsOut.ContinuousActions;
        continuousActions[0] = Input.GetAxisRaw("Horizontal");
        continuousActions[1] = Input.GetAxisRaw("Vertical");
    }

    private void OnCollisionEnter(Collision collision)
    {
        if (!collision.gameObject.CompareTag("Ball")) return;
        if (envController != null)
        {
            // logging step only
            envController.RegisterTouch(this);
        }
    }
}

public enum Team
{
    Blue,
    Red
}