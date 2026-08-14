using UnityEngine;

// attach component to trigger collider representing the goal
[RequireComponent(typeof(Collider))]
public class GoalTrigger : MonoBehaviour
{
    [Tooltip("The environment controller managing this pitch.")]
    public SoccerEnvController envController;

    [Tooltip("The team credited when the ball enters this goal.")]
    public Team scoringTeam = Team.Blue;

    private void OnTriggerEnter(Collider other)
    {
        if (other.CompareTag("Ball"))
        {
            envController.GoalScored(scoringTeam);
        }
    }
}