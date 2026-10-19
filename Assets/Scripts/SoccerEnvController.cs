using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;

public class SoccerEnvController : MonoBehaviour
{
    private struct SpawnPose
    {
        public Vector3 Position;
        public Quaternion Rotation;
    }

    [Header("Team Setup")]
    public List<SoccerAgent> teamBlue;
    public List<SoccerAgent> teamRed;

    [Header("Field References")]
    public Transform ball;
    public Rigidbody ballRb;

    [Tooltip("Target goal defended by Red team")]
    public Transform redGoal;

    [Tooltip("Target goal defended by Blue team")]
    public Transform blueGoal;

    [Tooltip("Pitch boundaries reference for observation normalization")]
    public Collider pitchBoundsCollider;

    [Header("Episode Reset")]
    [Tooltip("Minimum Y coordinate threshold to end round")]
    public float ballFallThresholdY = -2f;

    [Tooltip("Maximum episode duration in seconds before forcing reset")]
    public float maxEpisodeLengthSeconds = 45f;

    [Tooltip("Randomized spawn offset radius for the ball")]
    public float ballSpawnJitterRadius = 2f;

    [Tooltip("Randomized spawn offset radius for agents")]
    public float agentSpawnJitterRadius = 1.5f;

    [Header("Reward Parameters")]
    [Tooltip("Group level reward delivered to the scoring team via MA-POCA")]
    public float goalScoredReward = 3.0f;
    public float goalConcededReward = -3.0f;

    [Tooltip("Group level shaping reward for advancing the ball toward opponent goal")]
    public float ballAdvanceShapingScale = 0.02f;
    public float maxBallAdvanceRewardPerStep = 0.002f;

    private float previousBallToRedGoalDistance;
    private float previousBallToBlueGoalDistance;

    private Team? lastTouchTeam;
    private SoccerAgent lastTouchAgent;

    private float episodeTimer;
    private Vector3 initialBallPosition;
    private readonly Dictionary<SoccerAgent, SpawnPose> defaultSpawns = new Dictionary<SoccerAgent, SpawnPose>();

    private SimpleMultiAgentGroup blueGroup;
    private SimpleMultiAgentGroup redGroup;

    // cache initialization state
    private bool referencesValid;

    private int totalGoalsInEpisode;
    private bool episodeEndedInGoal;
    private int ballTouchesInEpisode;
    private int passesCompletedInEpisode;
    private int interceptionsInEpisode;
    private StatsRecorder statsRecorder;
    private SoccerTraceRecorder traceRecorder;
    private int traceEpisode;

    private void Awake()
    {
        if (ball != null)
        {
            initialBallPosition = ball.position;
        }

        CacheSpawns(teamBlue);
        CacheSpawns(teamRed);

        blueGroup = new SimpleMultiAgentGroup();
        redGroup = new SimpleMultiAgentGroup();
        foreach (var agent in teamBlue) if (agent != null) blueGroup.RegisterAgent(agent);
        foreach (var agent in teamRed) if (agent != null) redGroup.RegisterAgent(agent);

        statsRecorder = Academy.Instance.StatsRecorder;
        traceRecorder = GetComponent<SoccerTraceRecorder>();

        referencesValid = ValidateReferences();
    }

    // startup check for required scene references to avoid per-frame check overhead.
    private bool ValidateReferences()
    {
        var missing = new List<string>();
        if (ball == null) missing.Add("ball");
        if (ballRb == null) missing.Add("ballRb");
        if (teamBlue == null || teamBlue.Count == 0 || teamBlue.Contains(null)) missing.Add("teamBlue (empty or contains a null slot)");
        if (teamRed == null || teamRed.Count == 0 || teamRed.Contains(null)) missing.Add("teamRed (empty or contains a null slot)");

        if (missing.Count > 0)
        {
            Debug.LogError($"[SoccerEnvController] Missing required references, disabling this controller: {string.Join(", ", missing)}. " +
                            "Check the Inspector on this GameObject — script reloads can reset serialized fields.");
            return false;
        }
        return true;
    }

    private void CacheSpawns(List<SoccerAgent> team)
    {
        foreach (var agent in team)
        {
            if (agent == null) continue;
            defaultSpawns[agent] = new SpawnPose
            {
                Position = agent.transform.position,
                Rotation = agent.transform.rotation
            };
        }
    }

    private void FixedUpdate()
    {
        if (!referencesValid) return;

        episodeTimer += Time.fixedDeltaTime;

        if (episodeTimer >= maxEpisodeLengthSeconds)
        {
            // max time limit reached + terminate episode without assigning team terminal rewards.
            ResetEnv();
            return;
        }

        if (ball.position.y < ballFallThresholdY)
        {
            // ball fell out of bounds
            ResetEnv();
            return;
        }

        ApplyBallAdvanceGroupReward();
        if (traceRecorder != null)
            traceRecorder.RecordPositions(traceEpisode, episodeTimer, ball.position, teamBlue, teamRed);
    }

    // distribute delta distance rewards through MA-POCA group channels to maintain proper credit assignment.
    private void ApplyBallAdvanceGroupReward()
    {
        if (ballAdvanceShapingScale <= 0f || redGoal == null || blueGoal == null) return;

        float currentToRed = Vector3.Distance(ball.position, redGoal.position);
        float blueDelta = previousBallToRedGoalDistance - currentToRed;
        float blueReward = Mathf.Clamp(blueDelta * ballAdvanceShapingScale, -maxBallAdvanceRewardPerStep, maxBallAdvanceRewardPerStep);
        if (blueReward != 0f)
        {
            blueGroup.AddGroupReward(blueReward);
            if (traceRecorder != null)
                traceRecorder.RecordTeamReward(traceEpisode, episodeTimer, Team.Blue, "ball_advance", blueReward, ball.position);
        }
        previousBallToRedGoalDistance = currentToRed;

        float currentToBlue = Vector3.Distance(ball.position, blueGoal.position);
        float redDelta = previousBallToBlueGoalDistance - currentToBlue;
        float redReward = Mathf.Clamp(redDelta * ballAdvanceShapingScale, -maxBallAdvanceRewardPerStep, maxBallAdvanceRewardPerStep);
        if (redReward != 0f)
        {
            redGroup.AddGroupReward(redReward);
            if (traceRecorder != null)
                traceRecorder.RecordTeamReward(traceEpisode, episodeTimer, Team.Red, "ball_advance", redReward, ball.position);
        }
        previousBallToBlueGoalDistance = currentToBlue;
    }

    // Reward telemetry is strictly optional and never changes training rewards.
    public void TraceAgentReward(SoccerAgent agent, string term, float value)
    {
        if (traceRecorder != null && traceRecorder.IsRecording && ball != null)
            traceRecorder.RecordAgentReward(traceEpisode, episodeTimer, agent, term, value, ball.position);
    }

    public void RegisterTouch(SoccerAgent agent)
    {
        // track passing and interception telemetry for TensorBoard
        bool isPass = lastTouchAgent != null && lastTouchAgent != agent && lastTouchAgent.team == agent.team;
        bool isInterception = lastTouchTeam != null && lastTouchTeam != agent.team;

        if (isPass) passesCompletedInEpisode++;
        else if (isInterception) interceptionsInEpisode++;

        lastTouchAgent = agent;
        lastTouchTeam = agent.team;
        ballTouchesInEpisode++;
    }

    public void GoalScored(Team scoringTeam)
    {
        totalGoalsInEpisode++;
        episodeEndedInGoal = true;

        var scoringGroup = scoringTeam == Team.Blue ? blueGroup : redGroup;
        var concedingGroup = scoringTeam == Team.Blue ? redGroup : blueGroup;

        scoringGroup.AddGroupReward(goalScoredReward);
        concedingGroup.AddGroupReward(goalConcededReward);
        if (traceRecorder != null && traceRecorder.IsRecording)
        {
            traceRecorder.RecordTeamReward(traceEpisode, episodeTimer, scoringTeam, "goal_scored", goalScoredReward, ball.position);
            traceRecorder.RecordTeamReward(traceEpisode, episodeTimer,
                scoringTeam == Team.Blue ? Team.Red : Team.Blue, "goal_conceded", goalConcededReward, ball.position);
        }

        ResetEnv();
    }

    public void ResetEnv()
    {
        if (traceRecorder != null)
            traceRecorder.RecordEpisodeEnd(traceEpisode, episodeTimer, totalGoalsInEpisode,
                ballTouchesInEpisode, passesCompletedInEpisode, interceptionsInEpisode, ball.position);
        traceEpisode++;

        if (statsRecorder != null)
        {
            statsRecorder.Add("Environment/GoalsPerEpisode", totalGoalsInEpisode, StatAggregationMethod.Average);
            statsRecorder.Add("Environment/EpisodeEndedInGoal", episodeEndedInGoal ? 1f : 0f, StatAggregationMethod.Average);
            statsRecorder.Add("Environment/BallTouchesPerEpisode", ballTouchesInEpisode, StatAggregationMethod.Average);
            statsRecorder.Add("Environment/PassesCompletedPerEpisode", passesCompletedInEpisode, StatAggregationMethod.Average);
            statsRecorder.Add("Environment/InterceptionsPerEpisode", interceptionsInEpisode, StatAggregationMethod.Average);
        }

        totalGoalsInEpisode = 0;
        episodeEndedInGoal = false;
        ballTouchesInEpisode = 0;
        passesCompletedInEpisode = 0;
        interceptionsInEpisode = 0;

        Vector3 jitter = new Vector3(Random.Range(-ballSpawnJitterRadius, ballSpawnJitterRadius), 0f,
            Random.Range(-ballSpawnJitterRadius, ballSpawnJitterRadius));
        ball.position = initialBallPosition + jitter;
        ballRb.linearVelocity = Vector3.zero;
        ballRb.angularVelocity = Vector3.zero;

        if (redGoal != null) previousBallToRedGoalDistance = Vector3.Distance(ball.position, redGoal.position);
        if (blueGoal != null) previousBallToBlueGoalDistance = Vector3.Distance(ball.position, blueGoal.position);

        episodeTimer = 0f;
        lastTouchTeam = null;
        lastTouchAgent = null;

        ResetTeamSpawns(teamBlue);
        ResetTeamSpawns(teamRed);

        // execute group level episode reset for MA-POCA compatibility.
        blueGroup.EndGroupEpisode();
        redGroup.EndGroupEpisode();
    }

    private void ResetTeamSpawns(List<SoccerAgent> team)
    {
        foreach (var agent in team)
        {
            if (agent == null) continue;
            if (!defaultSpawns.TryGetValue(agent, out SpawnPose pose)) continue;

            Vector3 jitter = new Vector3(Random.Range(-agentSpawnJitterRadius, agentSpawnJitterRadius), 0f,
                Random.Range(-agentSpawnJitterRadius, agentSpawnJitterRadius));
            agent.transform.SetPositionAndRotation(pose.Position + jitter, pose.Rotation);

            if (agent.TryGetComponent<Rigidbody>(out Rigidbody agentRb))
            {
                agentRb.linearVelocity = Vector3.zero;
                agentRb.angularVelocity = Vector3.zero;
            }
        }
    }
}