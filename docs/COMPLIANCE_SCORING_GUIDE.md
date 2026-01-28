# Compliance Scoring Guide for Pharmacies

## What is Compliance Scoring?

Compliance scoring is a **transparent, organization-level metric** that measures how well your pharmacy follows required safety and reporting rules over time.

Think of it like:
- A credit score, but for regulatory discipline
- An Uber driver rating, but for pharmacovigilance behavior
- A quality score that shows trustworthiness

---

## Why Does It Matter?

### For Your Pharmacy
- Shows your commitment to safety
- Builds trust with pharmaceutical companies
- Unlocks advanced features
- Improves your reputation

### For Pharmaceutical Companies
- Identifies reliable data sources
- Prioritizes follow-ups from high-compliance pharmacies
- Detects unreliable data sources
- Makes better business decisions

### For Regulators
- Demonstrates transparent governance
- Shows data quality assurance
- Proves accountability
- Supports regulatory oversight

---

## Scoring System

### Score Range: 0 – 100

```
🟢 Compliant (80–100)
   ✓ All submissions on time
   ✓ Alerts acknowledged promptly
   ✓ Data quality excellent
   ✓ No schema violations

🟡 Attention Required (60–79)
   ⚠ Some late submissions
   ⚠ Occasional schema violations
   ⚠ Needs improvement
   ⚠ Some alerts not acknowledged

🔴 Non-compliant (<60)
   ✗ Frequent late submissions
   ✗ Repeated schema violations
   ✗ Requires intervention
   ✗ Ignored alerts
```

---

## What Gets Scored?

### ✅ Positive Behaviors (Increase Score)

| Behavior | Points | Frequency |
|----------|--------|-----------|
| On-time ADR submission | +5 | Per submission |
| Alert acknowledged within 24 hrs | +3 | Per alert |
| Severe ADR reported within SLA | +10 | Per severe case |
| Consistent reporting frequency | +2 | Monthly |
| Correct Excel schema | +2 | Per upload |
| Complete required fields | +1 | Per submission |

**Example**: Submit 10 reports on time = +50 points

### ❌ Negative Behaviors (Decrease Score)

| Behavior | Points | Frequency |
|----------|--------|-----------|
| Late submission | -5 | Per late submission |
| Ignored alert | -10 | Per ignored alert |
| Invalid Excel upload | -2 | Per invalid upload |
| Missing mandatory fields | -3 | Per submission |
| Long gaps in reporting | -5 | Per month without submission |
| Schema violation | -2 | Per violation |

**Example**: 2 late submissions + 1 ignored alert = -20 points

---

## How Scoring Works

### Starting Point
- New pharmacies: 100 points (perfect score)
- Existing pharmacies: Current score

### Automatic Updates
- Score updates automatically
- No manual review needed
- Real-time calculation
- Updated daily

### Score Calculation
```
Current Score = Base Score + Positive Actions - Negative Actions

Example:
100 (base) + 15 (5 on-time submissions) - 10 (1 ignored alert) = 105
(capped at 100)
```

---

## Viewing Your Score

### In Dashboard
1. Go to **Pharmacy Dashboard**
2. Look for **Compliance Status** card
3. See your score and status

### In Reports
1. Go to **Reports**
2. Click **Compliance Score**
3. View detailed breakdown

### API Endpoint
```
GET /api/pharmacy/reports/compliance-score

Response:
{
    "compliance_score": 85,
    "status": "Compliant",
    "status_color": "green",
    "last_updated": "2024-01-28T19:34:15"
}
```

---

## Improving Your Score

### Quick Wins (Easy)
1. **Submit on time**: +5 per submission
   - Set calendar reminders
   - Use automated workflows

2. **Acknowledge alerts**: +3 per alert
   - Check alerts daily
   - Respond within 24 hours

3. **Use correct schema**: +2 per upload
   - Download fresh template
   - Follow column names exactly

### Medium Effort
4. **Complete all fields**: +1 per submission
   - Don't leave blanks
   - Use "unknown" if unsure

5. **Report severe cases quickly**: +10 per case
   - Flag serious reactions
   - Report within SLA

### Long-term Strategy
6. **Consistent reporting**: +2 per month
   - Report regularly
   - Avoid gaps

7. **Data quality**: Implicit in all actions
   - Accurate information
   - Proper categorization

---

## Score Scenarios

### Scenario 1: New Pharmacy
```
Starting Score: 100

Month 1:
- 5 on-time submissions: +25
- 2 alerts acknowledged: +6
- 1 invalid Excel: -2
- Total: 100 + 25 + 6 - 2 = 129 (capped at 100)

Status: 🟢 Compliant
```

### Scenario 2: Struggling Pharmacy
```
Starting Score: 75

Month 1:
- 2 late submissions: -10
- 1 ignored alert: -10
- 1 schema violation: -2
- Total: 75 - 10 - 10 - 2 = 53

Status: 🔴 Non-compliant
Action: Pharmacy receives improvement notice
```

### Scenario 3: Recovering Pharmacy
```
Starting Score: 55

Month 1:
- 8 on-time submissions: +40
- 3 alerts acknowledged: +9
- Correct schema: +2
- Total: 55 + 40 + 9 + 2 = 106 (capped at 100)

Status: 🟢 Compliant
Action: Pharmacy removed from watch list
```

---

## What Happens at Different Scores?

### 🟢 Compliant (80–100)
**Benefits**:
- ✓ Full access to all features
- ✓ Priority support
- ✓ Featured in "trusted pharmacies" list
- ✓ Better rates from pharma companies

**Responsibilities**:
- Maintain on-time submissions
- Acknowledge alerts promptly
- Keep data quality high

### 🟡 Attention Required (60–79)
**Warnings**:
- ⚠ Improvement notice sent
- ⚠ Monthly check-ins required
- ⚠ Some features restricted

**Actions**:
- Review submission process
- Improve data quality
- Acknowledge alerts faster

### 🔴 Non-compliant (<60)
**Consequences**:
- ✗ Limited feature access
- ✗ Mandatory compliance training
- ✗ Weekly check-ins required
- ✗ Possible suspension

**Recovery**:
- Work with compliance officer
- Implement improvement plan
- Demonstrate 30 days of compliance
- Request score review

---

## Transparency & Fairness

### How We Ensure Fairness

1. **Clear Rules**: All scoring rules published
2. **No Surprises**: You know what affects your score
3. **Automatic Calculation**: No manual bias
4. **Appeal Process**: Dispute incorrect scores
5. **Regular Review**: Score methodology updated annually

### Your Rights

- ✓ View your score anytime
- ✓ See detailed breakdown
- ✓ Understand each action
- ✓ Appeal disputed scores
- ✓ Request manual review

---

## Common Questions

### Q: Can my score go below 0?
**A**: No. Minimum score is 0. Maximum is 100.

### Q: How often is my score updated?
**A**: Daily. Changes reflect within 24 hours.

### Q: Can I reset my score?
**A**: No. But you can improve it through positive actions.

### Q: What if I disagree with my score?
**A**: Contact support with details. We'll review and explain.

### Q: Does my score affect my pharmacy's operations?
**A**: No. It's informational. But low scores may restrict features.

### Q: Can I see other pharmacies' scores?
**A**: No. Scores are private. Only you and pharma companies see yours.

### Q: What if I have a legitimate reason for a late submission?
**A**: Contact support. We may adjust the score if justified.

---

## Best Practices

### Daily
- [ ] Check for new alerts
- [ ] Acknowledge alerts within 24 hours
- [ ] Review pending submissions

### Weekly
- [ ] Review compliance status
- [ ] Check for any issues
- [ ] Plan upcoming submissions

### Monthly
- [ ] Review compliance score
- [ ] Analyze trends
- [ ] Plan improvements

### Quarterly
- [ ] Full compliance audit
- [ ] Update procedures
- [ ] Train staff

---

## Support & Resources

### Getting Help
- **Dashboard**: View your score anytime
- **Email**: compliance@inteleyzer.com
- **Phone**: +1-XXX-XXX-XXXX
- **Chat**: Available in dashboard

### Documentation
- Full guide: PHARMACY_REPORTS_MODULE.md
- Quick start: PHARMACY_REPORTS_QUICK_START.md
- API docs: API_DOCUMENTATION.md

### Training
- Video tutorials: Available in dashboard
- Webinars: Monthly compliance training
- Documentation: Comprehensive guides

---

## Key Principles

### 1. Transparency
- All rules are public
- No hidden scoring
- Clear explanations

### 2. Fairness
- Same rules for all pharmacies
- Automatic calculation
- Appeal process available

### 3. Improvement-Focused
- Score rewards good behavior
- Penalties are educational
- Recovery is always possible

### 4. Privacy
- Your score is private
- Only you and pharma companies see it
- No public rankings

### 5. Collaboration
- We want you to succeed
- Support available
- Improvement plans offered

---

## The Bottom Line

**Compliance scoring is about data reliability, not control.**

If done right:
- ✓ Pharma companies trust your data
- ✓ You improve behavior naturally
- ✓ Regulators respect the system
- ✓ Everyone benefits

---

**Last Updated**: January 28, 2024
**Version**: 1.0

For questions or feedback, contact: compliance@inteleyzer.com
