document.addEventListener('DOMContentLoaded', () => {
  const form = document.getElementById('loan-form');
  const submitBtn = document.getElementById('submitBtn');
  const btnText = submitBtn.querySelector('.btn-text');
  const loader = submitBtn.querySelector('.loader');
  
  const resultOverlay = document.getElementById('result-overlay');
  const closeResult = document.getElementById('closeResult');
  
  const statusIcon = document.getElementById('statusIcon');
  const resultTitle = document.getElementById('resultTitle');
  const approveBar = document.getElementById('approveBar');
  const rejectBar = document.getElementById('rejectBar');
  const approveValue = document.getElementById('approveValue');
  const rejectValue = document.getElementById('rejectValue');
  const resultMessage = document.getElementById('resultMessage');

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    // Set loading state
    submitBtn.classList.add('loading');
    btnText.classList.add('hidden');
    loader.classList.remove('hidden');
    
    const payload = {
      Age: parseInt(document.getElementById('age').value),
      Income: parseFloat(document.getElementById('income').value),
      LoanAmount: parseFloat(document.getElementById('loanAmount').value),
      CreditScore: parseInt(document.getElementById('creditScore').value),
      EmploymentYears: parseInt(document.getElementById('employmentYears').value)
    };

    try {
      const response = await fetch('/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(payload)
      });
      
      const data = await response.json();
      
      // Delay slightly for effect
      setTimeout(() => {
        showResult(data);
      }, 600);
      
    } catch (error) {
      console.error("Error connecting to API", error);
      alert("Failed to connect to the prediction backend. Make sure the FastAPI python server is running on port 8000.");
      
      // Reset button
      submitBtn.classList.remove('loading');
      btnText.classList.remove('hidden');
      loader.classList.add('hidden');
    }
  });

  function showResult(data) {
    // Reset button
    submitBtn.classList.remove('loading');
    btnText.classList.remove('hidden');
    loader.classList.add('hidden');
    
    // Reset classes
    statusIcon.className = 'status-icon';
    resultTitle.className = '';
    
    if (data.status === 'Approved') {
      statusIcon.classList.add('icon-approve');
      resultTitle.textContent = 'Application Approved';
      resultTitle.classList.add('status-approved');
      resultMessage.textContent = 'Congratulations! The AI model indicates high confidence in your repayment ability. You meet all NeoBank criteria.';
    } else {
      statusIcon.classList.add('icon-reject');
      resultTitle.textContent = 'Application Rejected';
      resultTitle.classList.add('status-rejected');
      resultMessage.textContent = 'Unfortunately, we cannot approve this loan request due to elevated risk parameters detected in your profile.';
    }
    
    // Show overlay
    resultOverlay.classList.remove('hidden');
    
    // Animate bars
    setTimeout(() => {
      approveBar.style.width = data.confidence_approve + '%';
      rejectBar.style.width = data.confidence_reject + '%';
      
      // Animate numbers
      animateValue(approveValue, 0, data.confidence_approve, 1000);
      animateValue(rejectValue, 0, data.confidence_reject, 1000);
    }, 100);
  }

  function animateValue(obj, start, end, duration) {
    let startTimestamp = null;
    const step = (timestamp) => {
      if (!startTimestamp) startTimestamp = timestamp;
      const progress = Math.min((timestamp - startTimestamp) / duration, 1);
      obj.innerHTML = (progress * (end - start) + start).toFixed(1) + '%';
      if (progress < 1) {
        window.requestAnimationFrame(step);
      }
    };
    window.requestAnimationFrame(step);
  }

  closeResult.addEventListener('click', () => {
    resultOverlay.classList.add('hidden');
    // reset bars
    approveBar.style.width = '0%';
    rejectBar.style.width = '0%';
  });
});
