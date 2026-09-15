// AI Tutor Application JavaScript

class AITutor {
    constructor() {
        this.loadingIndicator = document.getElementById('loadingIndicator');
        this.quizContainer = document.getElementById('quizContainer');
        this.quizContent = document.getElementById('quizContent');

        this.currentQuiz = null;
    }
    
    showLoading() {
        if (this.loadingIndicator) {
            this.loadingIndicator.style.display = 'flex';
            this.loadingIndicator.classList.add('show');
        }
    }

    hideLoading() {
        if (this.loadingIndicator) {
            this.loadingIndicator.style.display = 'none';
            this.loadingIndicator.classList.remove('show');
        }
    }

    hideChatInterface() {
        // Hide messages container
        const messagesContainer = document.getElementById('messages');
        if (messagesContainer) {
            messagesContainer.style.display = 'none';
        }
        
        // Hide chat input section
        const chatInputSection = document.querySelector('.chat-input');
        if (chatInputSection) {
            chatInputSection.style.display = 'none';
        }
        
        // Hide revision techniques panel - using correct ID selector
        const quickActions = document.getElementById('revisionTechniques');
        if (quickActions) {
            quickActions.style.display = 'none';
            console.log('📝 DEBUG: Hidden Quick Actions panel in hideChatInterface');
        }
    }

    showChatInterface() {
        // Show messages container
        const messagesContainer = document.getElementById('messages');
        if (messagesContainer) {
            messagesContainer.style.display = 'block';
        }
        
        // Only show chat input if Executive Summary has been generated
        // Check if there are any assistant messages (indicating summary exists)
        const assistantMessages = messagesContainer ? messagesContainer.querySelectorAll('.message.assistant') : [];
        const chatInputSection = document.querySelector('.chat-input');
        if (chatInputSection && assistantMessages.length > 0) {
            chatInputSection.style.display = 'block';
        }
        
        // Only show revision techniques panel if quiz is NOT active
        const quizContainer = document.getElementById('quizContainer');
        const isQuizActive = quizContainer && quizContainer.style.display === 'block';
        
        if (!isQuizActive) {
            const quickActions = document.getElementById('revisionTechniques');
            if (quickActions) {
                quickActions.style.display = 'block';
                console.log('📝 DEBUG: Showed Quick Actions panel in showChatInterface (quiz not active)');
            }
        } else {
            console.log('📝 DEBUG: Quiz is active, keeping Quick Actions hidden in showChatInterface');
        }
        
        // Hide calculation answer input if it exists
        const calcAnswerSection = document.getElementById('calculationAnswerInput');
        if (calcAnswerSection) {
            calcAnswerSection.style.display = 'none';
            console.log('📝 DEBUG: Hidden calculation answer input when showing chat interface');
        }
        
        // Hide calculation feedback actions if they exist
        const feedbackActionsSection = document.getElementById('calculationFeedbackActions');
        if (feedbackActionsSection) {
            feedbackActionsSection.style.display = 'none';
            console.log('📝 DEBUG: Hidden calculation feedback actions when showing chat interface');
        }

        // Hide the essay answer panel if it exists
        const essaySection = document.getElementById('essayAnswerInput');
        if (essaySection) essaySection.style.display = 'none';
    }

    // ---------------- Essay question practice (mirrors the calculation loop) ----------------
    showEssayAnswerInput() {
        const chatInputSection = document.querySelector('.chat-input');
        if (chatInputSection) chatInputSection.style.display = 'none';
        const quickActionsSection = document.getElementById('revisionTechniques');
        if (quickActionsSection) quickActionsSection.style.display = 'none';
        // Essay practice replaces any calculation practice UI
        const calcAnswerSection = document.getElementById('calculationAnswerInput');
        if (calcAnswerSection) calcAnswerSection.style.display = 'none';
        const feedbackActionsSection = document.getElementById('calculationFeedbackActions');
        if (feedbackActionsSection) feedbackActionsSection.style.display = 'none';

        const section = document.getElementById('essayAnswerInput');
        if (!section) return;
        section.style.display = 'block';
        const textarea = document.getElementById('essayAnswerText');
        const counter = document.getElementById('essayWordCount');
        if (textarea) {
            textarea.value = '';
            textarea.disabled = false;
            if (counter && !textarea.dataset.counterBound) {
                textarea.addEventListener('input', () => {
                    const words = textarea.value.trim() ? textarea.value.trim().split(/\s+/).length : 0;
                    counter.textContent = `${words} word${words === 1 ? '' : 's'}`;
                });
                textarea.dataset.counterBound = '1';
            }
            if (counter) counter.textContent = '0 words';
            textarea.focus();
        }
        const submitBtn = document.getElementById('essaySubmitBtn');
        if (submitBtn) submitBtn.disabled = false;
        section.scrollIntoView({ behavior: 'smooth', block: 'end' });
    }

    hideEssayAnswerInput() {
        const section = document.getElementById('essayAnswerInput');
        if (section) section.style.display = 'none';
        const messagesContainer = document.getElementById('messages');
        const assistantMessages = messagesContainer ? messagesContainer.querySelectorAll('.message.assistant') : [];
        const chatInputSection = document.querySelector('.chat-input');
        if (chatInputSection && assistantMessages.length > 0) chatInputSection.style.display = 'block';
        const revisionPanel = document.getElementById('revisionTechniques');
        if (revisionPanel) revisionPanel.style.display = 'block';
        const textarea = document.getElementById('essayAnswerText');
        if (textarea) textarea.value = '';
    }

    endEssayPractice() {
        this.hideEssayAnswerInput();
        fetch('/simple_chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: 'end practice' })
        })
        .then(response => parseJSONResponse(response))
        .then(result => { if (result.success) this.addMessage('assistant', result.response); })
        .catch(error => console.error('Error ending essay practice:', error));
    }

    nextEssayQuestion() {
        const textarea = document.getElementById('essayAnswerText');
        if (textarea) textarea.value = '';
        const counter = document.getElementById('essayWordCount');
        if (counter) counter.textContent = '0 words';
        if (typeof quickAction === 'function') quickAction('Essay question');
    }

    // Calculation Answer Input Management
    showCalculationAnswerInput() {
        console.log('📝 DEBUG: Starting showCalculationAnswerInput');
        
        // Hide general chat input
        const chatInputSection = document.querySelector('.chat-input');
        if (chatInputSection) {
            chatInputSection.style.display = 'none';
            console.log('📝 DEBUG: Hidden chat input section');
        }
        
        // Hide Quick Actions panel during calculation questions - using correct ID selector
        const quickActionsSection = document.getElementById('revisionTechniques');
        if (quickActionsSection) {
            quickActionsSection.style.display = 'none';
            console.log('📝 DEBUG: Hidden Quick Actions panel');
        }
        const essaySectionForCalc = document.getElementById('essayAnswerInput');
        if (essaySectionForCalc) essaySectionForCalc.style.display = 'none';
        
        // Find calculation answer input by ID
        let calcAnswerSection = document.getElementById('calculationAnswerInput');
        console.log('📝 DEBUG: Found calc answer section:', calcAnswerSection);
        
        if (!calcAnswerSection) {
            console.log('📝 DEBUG: Calc answer section not found, creating new one');
            // If not found, create it and insert in correct position
            calcAnswerSection = document.createElement('div');
            calcAnswerSection.id = 'calculationAnswerInput';
            calcAnswerSection.className = 'calculation-answer-input';
            calcAnswerSection.innerHTML = `
                <h5><i class="fas fa-calculator me-2"></i>Your Answer</h5>
                <div class="input-group">
                    <input type="text" id="calcAnswerInput" class="form-control" 
                           placeholder="Enter your numerical answer (e.g., 4.68, 15.2%, $250)..."
                           onkeypress="if(event.key==='Enter') submitCalculationAnswer()">
                    <button class="btn btn-success" id="calcSubmitBtn" onclick="submitCalculationAnswer()">
                        <i class="fas fa-check"></i> Check Answer
                    </button>
                </div>
                <div class="mt-2">
                    <button class="btn btn-outline-secondary btn-sm me-2" onclick="nextCalculationQuestion()">
                        <i class="fas fa-forward"></i> Next Question
                    </button>
                    <button class="btn btn-outline-danger btn-sm" onclick="endCalculationSession()">
                        <i class="fas fa-stop"></i> End Practice
                    </button>
                </div>
                <div id="calcAnswerLoading" style="display: none;" class="mt-2">
                    <div class="spinner-border spinner-border-sm" role="status">
                        <span class="visually-hidden">Loading...</span>
                    </div>
                    <span class="ms-2">Checking your answer...</span>
                </div>
            `;
            
            // Insert at end of main content (since quick actions are now hidden)
            const mainContent = document.querySelector('.col-md-8');
            if (mainContent) {
                mainContent.appendChild(calcAnswerSection);
                console.log('📝 DEBUG: Appended calc section to main content');
            }
        }
        
        if (calcAnswerSection) {
            calcAnswerSection.style.display = 'block';
            console.log('📝 DEBUG: Made calc answer section visible');
            
            // Ensure the input form is visible for new questions
            const calcAnswerInputGroup = document.querySelector('#calculationAnswerInput .input-group');
            if (calcAnswerInputGroup) {
                calcAnswerInputGroup.style.display = 'flex';
                console.log('📝 DEBUG: Made input form visible');
            }
            
            // Focus on the input field
            const calcInput = document.getElementById('calcAnswerInput');
            if (calcInput) {
                calcInput.focus();
                console.log('📝 DEBUG: Focused on calc input');
            }
        }
        
        console.log('📝 DEBUG: Completed showCalculationAnswerInput');
    }
    
    hideCalculationAnswerInput() {
        // Hide calculation answer input
        const calcAnswerSection = document.getElementById('calculationAnswerInput');
        if (calcAnswerSection) {
            calcAnswerSection.style.display = 'none';
        }
        
        // Hide feedback actions section too
        const feedbackActionsSection = document.getElementById('calculationFeedbackActions');
        if (feedbackActionsSection) {
            feedbackActionsSection.style.display = 'none';
        }
        
        // Only show general chat input if Executive Summary has been generated
        const messagesContainer = document.getElementById('messages');
        const assistantMessages = messagesContainer ? messagesContainer.querySelectorAll('.message.assistant') : [];
        const chatInputSection = document.querySelector('.chat-input');
        if (chatInputSection && assistantMessages.length > 0) {
            chatInputSection.style.display = 'block';
        }
        
        // Show Revision Techniques panel again
        const revisionPanel = document.getElementById('revisionTechniques');
        if (revisionPanel) {
            revisionPanel.style.display = 'block';
            console.log('Revision Techniques panel shown after calculation practice');
        }
        
        // Clear the calculation input
        const calcInput = document.getElementById('calcAnswerInput');
        if (calcInput) {
            calcInput.value = '';
        }
        
        console.log('Switched back to general chat input mode');
    }
    
    endCalculationPractice() {
        console.log('Calculation practice ending - showing Revision Techniques panel');
        
        // Immediately hide calculation interface and show Revision Techniques panel
        this.hideCalculationAnswerInput();
        
        // Send end command through chat for server-side cleanup
        fetch('/simple_chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                message: 'end practice'
            })
        })
        .then(response => parseJSONResponse(response))
        .then(result => {
            if (result.success) {
                this.addMessage('assistant', result.response);
            }
        })
        .catch(error => {
            console.error('Error ending calculation practice:', error);
            // UI is already cleaned up, just log the error
        });
    }
    
    nextCalculationQuestion() {
        console.log('Generating next calculation question');

        // Hide feedback actions
        const feedbackActionsSection = document.getElementById('calculationFeedbackActions');
        if (feedbackActionsSection) feedbackActionsSection.style.display = 'none';

        // Restore answer input and clear it
        const calcAnswerSection = document.getElementById('calculationAnswerInput');
        if (calcAnswerSection) calcAnswerSection.style.display = 'block';
        const calcInput = document.getElementById('calcAnswerInput');
        if (calcInput) calcInput.value = '';

        // Advance the equation index on the server, then generate the next question
        fetch('/increment_equation_index', { method: 'POST', headers: { 'Content-Type': 'application/json' } })
            .then(r => r.json())
            .then(data => {
                console.log('📝 Equation index advanced to', data.index, 'of', data.total);
                if (typeof quickAction === 'function') quickAction('Calculation questions');
            })
            .catch(error => {
                console.error('Failed to increment equation index:', error);
                if (typeof quickAction === 'function') quickAction('Calculation questions');
            });
    }
    
    async parseJSONResponse(response) {
        const contentType = response.headers.get('content-type');
        const responseText = await response.text();
        
        // Check if response is HTML (common when OpenAI returns error pages)
        if (contentType && contentType.includes('text/html') || responseText.trim().startsWith('<')) {
            console.error('Received HTML response instead of JSON:', responseText.substring(0, 200));
            throw new Error('Service temporarily unavailable. Please try again in a few moments.');
        }
        
        try {
            return JSON.parse(responseText);
        } catch (error) {
            console.error('JSON parsing error:', error);
            console.error('Response text:', responseText.substring(0, 200));
            // If it's an HTML response that wasn't caught above
            if (responseText.includes('<html>') || responseText.includes('<!DOCTYPE')) {
                throw new Error('Service temporarily unavailable. Please try again in a few moments.');
            }
            throw new Error('Invalid response format. Please try again.');
        }
    }
    
    // Highlight state for the Revision Techniques buttons. Mirrors the
    // processing/active class handling in index.html's quickAction() so the
    // quiz and infographic buttons behave like the streaming ones: green
    // while working ('processing'), green when done ('active'), cleared on
    // failure or when another technique starts.
    setRevisionButtonState(onclickMarker, state) {
        const buttons = document.querySelectorAll('.quick-actions .btn-outline-primary');
        buttons.forEach(btn => btn.classList.remove('processing', 'active'));
        if (!onclickMarker || !state) return;
        const target = Array.from(buttons).find(btn =>
            (btn.getAttribute('onclick') || '').includes(onclickMarker));
        if (target) target.classList.add(state);
    }

    async startQuiz() {
        console.log('startQuiz called - using async polling pattern');
        this.setRevisionButtonState('startQuiz', 'processing');
        
        // Show loading with styled progress bar
        const messagesDiv = document.getElementById('messages');
        const progressDiv = document.createElement('div');
        progressDiv.className = 'message assistant';
        progressDiv.innerHTML = `
            <div class="message-content">
                <div class="d-flex align-items-center">
                    <div class="me-3">
                        <i class="fas fa-question-circle fa-2x text-primary"></i>
                    </div>
                    <div class="flex-grow-1">
                        <h6 class="mb-2">🧠 Generating Quiz...</h6>
                        <div class="progress">
                            <div class="progress-bar progress-bar-striped progress-bar-animated" 
                                 role="progressbar" style="width: 0%; background-color: #FF6600;" 
                                 aria-valuenow="0" aria-valuemin="0" aria-valuemax="100">
                                <span class="visually-hidden">Creating quiz questions...</span>
                            </div>
                        </div>
                        <small class="text-muted mt-1">
                            <i class="fas fa-brain me-1"></i>
                            Creating personalized quiz questions...
                        </small>
                    </div>
                </div>
            </div>
        `;
        messagesDiv.appendChild(progressDiv);
        
        // Animate progress bar
        const progressBar = progressDiv.querySelector('.progress-bar');
        let width = 0;
        const interval = setInterval(() => {
            width += 2;
            progressBar.style.width = width + '%';
            if (width >= 100) {
                clearInterval(interval);
                progressBar.style.backgroundColor = '#D6000D';
            }
        }, 1800); // 90 seconds total animation (50 steps * 1800ms)
        
        try {
            // Start the async quiz generation
            const response = await fetch('/start_quiz_generation', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                }
            });
            
            const data = await this.parseJSONResponse(response);
            
            if (response.ok && data.task_id) {
                console.log('✓ Quiz generation task started:', data.task_id);
                
                // Start polling for results
                this.startQuizPolling(data.task_id, progressDiv, interval);
            } else {
                console.error('Quiz generation start failed:', data.error);
                
                // Stop progress bar
                clearInterval(interval);
                
                // Remove progress bar
                if (messagesDiv.contains(progressDiv)) {
                    messagesDiv.removeChild(progressDiv);
                }
                
                // Restore Quick Actions panel on error
                const quickActionsSection = document.getElementById('revisionTechniques');
                if (quickActionsSection) {
                    quickActionsSection.style.display = 'block';
                    console.log('📝 DEBUG: Restored Quick Actions panel after quiz error');
                }
                
                this.setRevisionButtonState(null, null);
                this.addMessage('assistant', '❌ Quiz generation failed: ' + (data.error || 'Unknown error'));
            }
        } catch (error) {
            console.error('Quiz generation start error:', error);
            
            // Stop progress bar
            clearInterval(interval);
            
            // Remove progress bar if still there
            if (messagesDiv.contains(progressDiv)) {
                messagesDiv.removeChild(progressDiv);
            }
            
            // Restore Quick Actions panel on error
            const quickActionsSection = document.getElementById('revisionTechniques');
            if (quickActionsSection) {
                quickActionsSection.style.display = 'block';
                console.log('📝 DEBUG: Restored Quick Actions panel after quiz error');
            }
            
            this.setRevisionButtonState(null, null);
            this.addMessage('assistant', '❌ Quiz generation failed: ' + error.message);
        }
    }
    
    startQuizPolling(taskId, progressDiv, progressInterval) {
        console.log('🧠 Starting quiz polling for task:', taskId);
        
        const pollInterval = 2000; // 2 seconds between polls
        const maxAttempts = 45; // Maximum 45 attempts (90 seconds)
        let attempts = 0;
        
        const poll = () => {
            attempts++;
            console.log(`🧠 Quiz polling attempt ${attempts}/${maxAttempts} for task ${taskId}`);
            
            fetch(`/quiz_status/${taskId}`)
            .then(response => response.json())
            .then(data => {
                console.log('🧠 Quiz poll response:', data);
                
                if (data.status === 'complete' && data.success) {
                    console.log('✓ Quiz generation completed');
                    
                    // Stop progress bar
                    if (progressInterval) {
                        clearInterval(progressInterval);
                    }
                    
                    // Remove progress bar
                    const messagesDiv = document.getElementById('messages');
                    if (messagesDiv && messagesDiv.contains(progressDiv)) {
                        messagesDiv.removeChild(progressDiv);
                    }
                    
                    // Set up quiz with the generated questions
                    if (data.data && data.data.length > 0) {
                        this.currentQuiz = {
                            questions: data.data,
                            currentQuestionIndex: 0,
                            totalQuestions: data.data.length,
                            score: 0
                        };
                        console.log('📝 DEBUG: About to call showQuiz() - Quick Actions should disappear NOW');
                        this.setRevisionButtonState('startQuiz', 'active');
                        this.showQuiz();
                    } else {
                        this.setRevisionButtonState(null, null);
                        this.addMessage('assistant', '❌ No quiz questions were generated. Please try again.');
                    }
                    
                } else if (data.status === 'failed' || data.status === 'error') {
                    console.error('✗ Quiz generation failed:', data.error);
                    
                    // Stop progress bar
                    if (progressInterval) {
                        clearInterval(progressInterval);
                    }
                    
                    // Remove progress bar
                    const messagesDiv = document.getElementById('messages');
                    if (messagesDiv && messagesDiv.contains(progressDiv)) {
                        messagesDiv.removeChild(progressDiv);
                    }
                    
                    this.setRevisionButtonState(null, null);
                    this.addMessage('assistant', '❌ ' + (data.error || 'Quiz generation failed'));
                    
                } else if (data.status === 'pending' || data.status === 'running') {
                    // Task still running, continue polling
                    if (attempts < maxAttempts) {
                        setTimeout(poll, pollInterval);
                    } else {
                        console.error('✗ Quiz polling timeout');
                        
                        // Stop progress bar
                        if (progressInterval) {
                            clearInterval(progressInterval);
                        }
                        
                        // Remove progress bar
                        const messagesDiv = document.getElementById('messages');
                        if (messagesDiv && messagesDiv.contains(progressDiv)) {
                            messagesDiv.removeChild(progressDiv);
                        }
                        
                        this.setRevisionButtonState(null, null);
                        this.addMessage('assistant', '❌ Quiz generation is taking longer than expected. Please try again.');
                    }
                } else {
                    console.error('✗ Unknown quiz task status:', data.status);
                    
                    // Stop progress bar
                    if (progressInterval) {
                        clearInterval(progressInterval);
                    }
                    
                    // Remove progress bar
                    const messagesDiv = document.getElementById('messages');
                    if (messagesDiv && messagesDiv.contains(progressDiv)) {
                        messagesDiv.removeChild(progressDiv);
                    }
                    
                    this.setRevisionButtonState(null, null);
                    this.addMessage('assistant', '❌ Unknown error occurred during quiz generation');
                }
            })
            .catch(error => {
                console.error('✗ Quiz polling error:', error);
                
                if (attempts < maxAttempts) {
                    // Retry on network error
                    setTimeout(poll, pollInterval);
                } else {
                    // Stop progress bar
                    if (progressInterval) {
                        clearInterval(progressInterval);
                    }
                    
                    // Remove progress bar
                    const messagesDiv = document.getElementById('messages');
                    if (messagesDiv && messagesDiv.contains(progressDiv)) {
                        messagesDiv.removeChild(progressDiv);
                    }
                    
                    this.setRevisionButtonState(null, null);
                    this.addMessage('assistant', '❌ Network error during quiz generation. Please try again.');
                }
            });
        };
        
        // Start polling immediately
        poll();
    }
    
    showQuiz() {
        console.log('Quiz appearing - hiding Revision Techniques panel');
        
        // Show quiz container
        if (this.quizContainer) {
            this.quizContainer.style.display = 'block';
        }
        
        // Hide chat input during quiz
        const chatInput = document.querySelector('.chat-input');
        if (chatInput) {
            chatInput.style.display = 'none';
        }
        
        // Hide Revision Techniques panel during quiz
        const revisionPanel = document.getElementById('revisionTechniques');
        if (revisionPanel) {
            revisionPanel.style.display = 'none';
            console.log('Revision Techniques panel hidden for quiz');
        }
        
        this.displayQuizQuestion();
    }
    
    displayQuizQuestion() {
        console.log('displayQuizQuestion called with async data');
        console.log('Current quiz:', this.currentQuiz);
        console.log('Quiz content element:', this.quizContent);
        
        // Get current question from the questions array
        const question = this.currentQuiz.questions[this.currentQuiz.currentQuestionIndex];
        const questionNumber = this.currentQuiz.currentQuestionIndex + 1;
        const progress = (questionNumber / this.currentQuiz.totalQuestions) * 100;
        
        console.log('Question:', question);
        console.log('Question number:', questionNumber);
        console.log('Progress:', progress);
        
        // Debug question structure
        if (!question) {
            console.error('Question is undefined or null');
            return;
        }
        
        console.log('Question.question:', question.question);
        console.log('Question.options:', question.options);
        console.log('Question object keys:', Object.keys(question));
        
        // Validate question structure
        if (!question.question) {
            console.error('Question.question is undefined');
            this.addMessage('assistant', '❌ Quiz error: Question text is missing');
            return;
        }
        
        if (!question.options || !Array.isArray(question.options)) {
            console.error('Question.options is not an array:', question.options);
            this.addMessage('assistant', '❌ Quiz error: Question options are malformed');
            return;
        }
        
        if (this.quizContent) {
            // Clear existing content
            this.quizContent.innerHTML = '';
            
            // Create safe DOM structure
            const progressIndicator = document.createElement('div');
            progressIndicator.className = 'progress-indicator';
            
            const progressText = document.createElement('div');
            progressText.className = 'progress-text d-flex justify-content-between align-items-center';
            progressText.style.padding = '0 1rem 1rem 1rem'; // Increased bottom padding for more space
            progressText.style.marginBottom = '1rem'; // Increased bottom margin for more space between question/score and content
            
            const questionSpan = document.createElement('span');
            questionSpan.textContent = `Question ${questionNumber} of ${this.currentQuiz.totalQuestions}`;
            questionSpan.style.fontWeight = 'bold';
            
            const scoreSpan = document.createElement('span');
            scoreSpan.textContent = `Score: ${this.currentQuiz.score}`;
            scoreSpan.style.fontWeight = 'bold';
            
            progressText.appendChild(questionSpan);
            progressText.appendChild(scoreSpan);
            
            const progressBar = document.createElement('div');
            progressBar.className = 'progress';
            
            // Create progress bar safely using DOM methods
            const progressBarInner = document.createElement('div');
            progressBarInner.className = 'progress-bar';
            progressBarInner.style.width = `${progress}%`;
            progressBarInner.style.backgroundColor = progress >= 100 ? '#D6000D' : '#FF6600';
            
            progressBar.appendChild(progressBarInner);
            
            progressIndicator.appendChild(progressText);
            progressIndicator.appendChild(progressBar);
            
            // Create question section
            const quizQuestion = document.createElement('div');
            quizQuestion.className = 'quiz-question';
            
            const questionHeader = document.createElement('h5');
            questionHeader.textContent = question.question; // Safe: uses textContent
            
            const quizOptions = document.createElement('div');
            quizOptions.className = 'quiz-options';
            
            // Create options safely
            question.options.forEach((option, index) => {
                const optionDiv = document.createElement('div');
                optionDiv.className = 'form-check quiz-option';

                const input = document.createElement('input');
                input.type = 'radio';
                input.name = 'answer';
                input.className = 'form-check-input';
                input.value = option; // Note: form values need special handling if they contain user data
                input.id = `option${index}`;

                const label = document.createElement('label');
                label.setAttribute('for', `option${index}`);
                label.className = 'form-check-label';
                label.textContent = option; // Safe: uses textContent
                
                optionDiv.appendChild(input);
                optionDiv.appendChild(label);
                quizOptions.appendChild(optionDiv);
            });
            
            quizQuestion.appendChild(questionHeader);
            quizQuestion.appendChild(quizOptions);
            
            // Create button section
            const buttonDiv = document.createElement('div');
            buttonDiv.className = 'd-flex justify-content-between';
            
            const endButton = document.createElement('button');
            endButton.className = 'btn btn-secondary';
            endButton.textContent = 'End Quiz';
            endButton.onclick = () => this.endQuiz();
            
            const submitButton = document.createElement('button');
            submitButton.className = 'btn btn-primary';
            submitButton.textContent = 'Submit Answer';
            submitButton.disabled = false; // Ensure button is enabled for new question
            submitButton.onclick = () => this.submitQuizAnswer();
            
            buttonDiv.appendChild(endButton);
            buttonDiv.appendChild(submitButton);
            
            // Append all elements
            this.quizContent.appendChild(progressIndicator);
            this.quizContent.appendChild(quizQuestion);
            this.quizContent.appendChild(buttonDiv);
            console.log('Quiz content populated with async data');
        } else {
            console.error('Quiz content element not found!');
        }
    }
    
    async submitQuizAnswer() {
        const selectedAnswer = document.querySelector('input[name="answer"]:checked');
        if (!selectedAnswer) {
            alert('Please select an answer.');
            return;
        }
        
        // Prevent multiple submissions to the same question
        const submitButton = document.querySelector('.btn-primary');
        if (submitButton && submitButton.disabled) {
            console.log('Answer already submitted for this question');
            return;
        }
        
        // Disable submit button and all radio buttons to prevent multiple submissions
        if (submitButton && submitButton.textContent === 'Submit Answer') {
            submitButton.disabled = true;
            submitButton.textContent = 'Answer Submitted';
            submitButton.className = 'btn btn-secondary';
        }
        
        // Disable all radio buttons to prevent changing selection after submission
        const allRadioButtons = document.querySelectorAll('input[name="answer"]');
        allRadioButtons.forEach(radio => {
            radio.disabled = true;
        });
        
        // Get current question (capture it before any changes)
        const currentQuestionIndex = this.currentQuiz.currentQuestionIndex;
        const currentQuestion = this.currentQuiz.questions[currentQuestionIndex];
        const userAnswer = selectedAnswer.value;
        const correctAnswer = currentQuestion.correct_answer;
        const isCorrect = userAnswer.trim() === correctAnswer.trim();
        
        console.log('Question index:', currentQuestionIndex);
        console.log('User answer:', userAnswer);
        console.log('Correct answer:', correctAnswer);
        console.log('Is correct:', isCorrect);
        
        // Update score immediately if correct
        if (isCorrect) {
            this.currentQuiz.score++;
            
            // Update score display immediately
            const scoreSpan = document.querySelector('.progress-text span:last-child');
            if (scoreSpan) {
                scoreSpan.textContent = `Score: ${this.currentQuiz.score}`;
                console.log('Updated score display immediately to:', this.currentQuiz.score);
            }
        }
        
        // Show colored feedback box immediately
        this.showQuizFeedbackBox(isCorrect);
        
        // Show result immediately (client-side processing)
        this.showQuizResult({
            correct: isCorrect,
            explanation: currentQuestion.explanation,
            quiz_complete: currentQuestionIndex >= this.currentQuiz.totalQuestions - 1,
            final_score: this.currentQuiz.score,
            total_questions: this.currentQuiz.totalQuestions
        });
    }
    
    showQuizFeedbackBox(isCorrect) {
        // Create colored feedback box that appears immediately
        const feedbackBox = document.createElement('div');
        feedbackBox.className = 'quiz-feedback-box';
        
        // Apply styling for immediate colored feedback
        feedbackBox.style.cssText = `
            margin: 1rem 0 !important;
            padding: 0.75rem 1rem !important;
            border-radius: 8px !important;
            border: 2px solid ${isCorrect ? '#28a745' : '#dc3545'} !important;
            background-color: ${isCorrect ? '#d4edda' : '#f8d7da'} !important;
            color: ${isCorrect ? '#155724' : '#721c24'} !important;
            display: block !important;
            width: 100% !important;
            box-sizing: border-box !important;
            font-weight: bold !important;
            text-align: center !important;
            font-size: 1.1rem !important;
        `;
        
        feedbackBox.textContent = isCorrect ? 'Correct!' : 'Incorrect!';
        
        // Insert feedback box right after the options
        const optionsDiv = document.querySelector('.quiz-options');
        if (optionsDiv && optionsDiv.parentNode) {
            optionsDiv.parentNode.insertBefore(feedbackBox, optionsDiv.nextSibling);
        } else {
            // Fallback: add to quiz content
            this.quizContent.appendChild(feedbackBox);
        }
        
        console.log('Added immediate feedback box:', isCorrect ? 'Correct!' : 'Incorrect!');
    }
    
    showQuizResult(data) {
        console.log('Quiz result - correct:', data.correct);
        
        // Show detailed result message (explanation and correct answer)
        const resultDiv = document.createElement('div');
        resultDiv.className = 'quiz-result-details';
        resultDiv.style.cssText = `
            margin: 1rem 0 !important;
            padding: 1rem !important;
            border-radius: 8px !important;
            border: 1px solid #dee2e6 !important;
            background-color: #f8f9fa !important;
            color: #495057 !important;
            display: block !important;
            width: 100% !important;
            box-sizing: border-box !important;
        `;
        
        // Add correct answer display
        const answerPara = document.createElement('p');
        answerPara.className = 'quiz-answer';
        answerPara.style.marginBottom = '0.5rem';
        answerPara.style.fontWeight = 'bold';
        
        if (!data.correct) {
            answerPara.textContent = `The correct answer is: ${this.currentQuiz.questions[this.currentQuiz.currentQuestionIndex].correct_answer}`;
        } else {
            answerPara.textContent = `You selected the correct answer: ${this.currentQuiz.questions[this.currentQuiz.currentQuestionIndex].correct_answer}`;
        }
        
        resultDiv.appendChild(answerPara);
        
        // Show explanation if available
        if (data.explanation && data.explanation.trim()) {
            const explanationPara = document.createElement('p');
            explanationPara.className = 'quiz-explanation';
            explanationPara.style.marginBottom = '0';
            explanationPara.textContent = data.explanation;
            resultDiv.appendChild(explanationPara);
        }
        
        this.quizContent.appendChild(resultDiv);

        if (data.quiz_complete) {
            const scoreDiv = document.createElement('div');
            scoreDiv.className = 'score-display';
            scoreDiv.textContent = `Quiz Complete! Final Score: ${data.final_score}/${data.total_questions}`;
            this.quizContent.appendChild(scoreDiv);
            
            const closeButton = document.createElement('button');
            closeButton.className = 'btn btn-primary';
            closeButton.textContent = 'Close Quiz';
            closeButton.onclick = () => aiTutor.endQuiz();
            this.quizContent.appendChild(closeButton);
        } else {
            // Move to next question
            this.currentQuiz.currentQuestionIndex++;
            
            // Show next question button using safe DOM methods
            const buttonContainer = document.createElement('div');
            buttonContainer.className = 'd-flex justify-content-center mt-3';
            
            const nextButton = document.createElement('button');
            nextButton.className = 'btn btn-primary';
            nextButton.textContent = 'Next Question';
            nextButton.onclick = () => aiTutor.showNextQuestion();
            
            buttonContainer.appendChild(nextButton);
            this.quizContent.appendChild(buttonContainer);
        }
    }
    
    showNextQuestion() {
        console.log('Moving to next question:', this.currentQuiz.currentQuestionIndex);
        this.displayQuizQuestion();
    }
    
    endQuiz() {
        console.log('Quiz ending - showing Revision Techniques panel');
        
        // Hide quiz container
        if (this.quizContainer) {
            this.quizContainer.style.display = 'none';
        }
        this.currentQuiz = null;
        
        // Show Revision Techniques panel again
        const revisionPanel = document.getElementById('revisionTechniques');
        if (revisionPanel) {
            revisionPanel.style.display = 'block';
            console.log('Revision Techniques panel shown after quiz');
        }
        
        // Show chat input again if executive summary exists
        const messagesContainer = document.getElementById('messages');
        const assistantMessages = messagesContainer ? messagesContainer.querySelectorAll('.message.assistant') : [];
        const chatInput = document.querySelector('.chat-input');
        if (chatInput && assistantMessages.length > 0) {
            chatInput.style.display = 'block';
        }
        
        // Add completion message
        this.addMessage('assistant', '🎯 Quiz completed! You can start another quiz or ask me any questions about your study material.');
    }
    
    async startInfographic() {
        console.log('startInfographic called - using async polling pattern');
        this.setRevisionButtonState('startInfographic', 'processing');

        // Show loading with styled progress bar
        const messagesDiv = document.getElementById('messages');
        const progressDiv = document.createElement('div');
        progressDiv.className = 'message assistant';
        progressDiv.innerHTML = `
            <div class="message-content">
                <div class="d-flex align-items-center">
                    <div class="me-3">
                        <i class="fas fa-image fa-2x text-primary"></i>
                    </div>
                    <div class="flex-grow-1">
                        <h6 class="mb-2">🎨 Creating your revision infographic</h6>
                        <div class="progress">
                            <div class="progress-bar progress-bar-striped progress-bar-animated"
                                 role="progressbar" style="width: 0%; background-color: #FF6600;"
                                 aria-valuenow="0" aria-valuemin="0" aria-valuemax="100">
                                <span class="visually-hidden">Creating revision infographic...</span>
                            </div>
                        </div>
                        <small class="text-muted mt-1">
                            <i class="fas fa-paint-brush me-1"></i>
                            This takes a few minutes (we check it before showing it). No need to wait: enter your email below and we will send it as a PDF.
                        </small>
                    </div>
                </div>
            </div>
        `;
        messagesDiv.appendChild(progressDiv);
        progressDiv.scrollIntoView({ behavior: 'smooth', block: 'end' });

        // Animate progress bar towards (but never past) 95% over ~6 minutes
        const progressBar = progressDiv.querySelector('.progress-bar');
        let width = 0;
        const interval = setInterval(() => {
            width += 1;
            progressBar.style.width = width + '%';
            if (width >= 95) {
                clearInterval(interval);
            }
        }, 4000);

        try {
            const response = await fetch('/start_infographic_generation', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                }
            });

            const data = await this.parseJSONResponse(response);

            if (response.ok && data.task_id) {
                console.log('✓ Infographic generation task started:', data.task_id);
                this.lastInfographicTaskId = data.task_id;
                this.attachInfographicEmailForm(progressDiv.querySelector('.flex-grow-1'), data.task_id, {
                    label: 'Email me the infographic PDF when it is ready',
                    buttonText: 'Email me'
                });
                this.startInfographicPolling(data.task_id, progressDiv, interval);
            } else {
                console.error('Infographic generation start failed:', data.error);
                clearInterval(interval);
                if (messagesDiv.contains(progressDiv)) {
                    messagesDiv.removeChild(progressDiv);
                }
                this.setRevisionButtonState(null, null);
                this.addMessage('assistant', '❌ Infographic generation failed: ' + (data.error || 'Unknown error'));
            }
        } catch (error) {
            console.error('Infographic generation start error:', error);
            clearInterval(interval);
            if (messagesDiv.contains(progressDiv)) {
                messagesDiv.removeChild(progressDiv);
            }
            this.setRevisionButtonState(null, null);
            this.addMessage('assistant', '❌ Infographic generation failed: ' + error.message);
        }
    }

    startInfographicPolling(taskId, progressDiv, progressInterval) {
        console.log('🎨 Starting infographic polling for task:', taskId);

        const pollInterval = 3000; // 3 seconds between polls
        const maxAttempts = 240;   // up to 12 minutes (generation + check/correction rounds)
        let attempts = 0;

        const cleanupProgress = () => {
            if (progressInterval) {
                clearInterval(progressInterval);
            }
            const messagesDiv = document.getElementById('messages');
            if (messagesDiv && messagesDiv.contains(progressDiv)) {
                messagesDiv.removeChild(progressDiv);
            }
        };

        const poll = () => {
            attempts++;
            console.log(`🎨 Infographic polling attempt ${attempts}/${maxAttempts} for task ${taskId}`);

            fetch(`/infographic_status/${taskId}`)
            .then(response => response.json())
            .then(data => {
                if (data.status === 'complete' && data.success) {
                    console.log('✓ Infographic generation completed');
                    cleanupProgress();

                    if (data.data) {
                        this.setRevisionButtonState('startInfographic', 'active');
                        this.showInfographic(data.data);
                    } else {
                        this.setRevisionButtonState(null, null);
                        this.addMessage('assistant', '❌ No infographic was generated. Please try again.');
                    }

                } else if (data.status === 'failed' || data.status === 'error') {
                    console.error('✗ Infographic generation failed:', data.error);
                    cleanupProgress();
                    this.setRevisionButtonState(null, null);
                    this.addMessage('assistant', '❌ ' + (data.error || 'Infographic generation failed'));

                } else if (data.status === 'pending' || data.status === 'running') {
                    if (attempts < maxAttempts) {
                        setTimeout(poll, pollInterval);
                    } else {
                        console.error('✗ Infographic polling timeout');
                        cleanupProgress();
                        this.setRevisionButtonState(null, null);
                        this.addMessage('assistant', '❌ Infographic generation is taking longer than expected. Please try again.');
                    }
                } else {
                    console.error('✗ Unknown infographic task status:', data.status);
                    cleanupProgress();
                    this.setRevisionButtonState(null, null);
                    this.addMessage('assistant', '❌ Unknown error occurred during infographic generation');
                }
            })
            .catch(error => {
                console.error('✗ Infographic polling error:', error);
                if (attempts < maxAttempts) {
                    setTimeout(poll, pollInterval);
                } else {
                    cleanupProgress();
                    this.setRevisionButtonState(null, null);
                    this.addMessage('assistant', '❌ Network error during infographic generation. Please try again.');
                }
            });
        };

        poll();
    }

    showInfographic(imageB64) {
        // Keep the image on the instance so the viewer/download handlers do
        // not need multi-megabyte inline onclick attributes.
        this.infographicB64 = imageB64;

        const messagesDiv = document.getElementById('messages');
        const msgDiv = document.createElement('div');
        msgDiv.className = 'message assistant';
        msgDiv.innerHTML = `
            <div class="message-content">
                <h6 class="mb-1">🎨 Your revision infographic is ready!</h6>
                <p class="text-muted small mb-2">Click the image (or "View &amp; Zoom") to open it full screen — scroll, pinch or use the buttons to zoom in on the small text.</p>
                <img class="infographic-preview" alt="Revision guide infographic">
                <div class="mt-2">
                    <button class="btn btn-sm btn-primary me-2 infographic-view-btn" style="background-color: #D6000D; border-color: #D6000D;">
                        <i class="fas fa-search-plus me-1"></i>View &amp; Zoom
                    </button>
                    <button class="btn btn-sm btn-outline-secondary me-2 infographic-download-btn">
                        <i class="fas fa-download me-1"></i>Download PNG
                    </button>
                    <button class="btn btn-sm btn-outline-secondary infographic-email-btn" hidden>
                        <i class="fas fa-envelope me-1"></i>Email as PDF
                    </button>
                </div>
                <div class="infographic-email-slot"></div>
            </div>
        `;
        msgDiv.querySelector('.infographic-preview').src = 'data:image/png;base64,' + imageB64;
        msgDiv.querySelector('.infographic-preview').addEventListener('click', () => this.openInfographicViewer());
        msgDiv.querySelector('.infographic-view-btn').addEventListener('click', () => this.openInfographicViewer());
        msgDiv.querySelector('.infographic-download-btn').addEventListener('click', () => this.downloadInfographic());
        const emailBtn = msgDiv.querySelector('.infographic-email-btn');
        if (this.infographicEmailEnabled()) {
            emailBtn.hidden = false;
            emailBtn.addEventListener('click', () => {
                const slot = msgDiv.querySelector('.infographic-email-slot');
                if (slot.childElementCount) { slot.innerHTML = ''; return; }
                this.attachInfographicEmailForm(slot, this.lastInfographicTaskId, {
                    label: 'Send this infographic to my email as a PDF',
                    buttonText: 'Send PDF',
                    includeImage: true
                });
                slot.querySelector('input').focus();
            });
        }
        messagesDiv.appendChild(msgDiv);
        msgDiv.scrollIntoView({ behavior: 'smooth', block: 'end' });
    }

    infographicEmailEnabled() {
        return !!(window.INFOGRAPHIC_EMAIL && window.INFOGRAPHIC_EMAIL.enabled);
    }

    // Small inline form: [email] [button] + status line. Posts to /email_infographic.
    attachInfographicEmailForm(container, taskId, opts = {}) {
        if (!container || !this.infographicEmailEnabled()) return;
        const prefill = (window.INFOGRAPHIC_EMAIL && window.INFOGRAPHIC_EMAIL.prefill) || '';
        const form = document.createElement('form');
        form.className = 'infographic-email-form mt-2';
        form.innerHTML = `
            <label class="form-label small mb-1"><i class="fas fa-envelope me-1"></i>${opts.label || 'Email me a PDF copy'}</label>
            <div class="input-group input-group-sm">
                <input type="email" class="form-control" placeholder="you@qub.ac.uk" required autocomplete="email">
                <button class="btn btn-outline-secondary" type="submit">${opts.buttonText || 'Send'}</button>
            </div>
            <div class="form-text infographic-email-status"></div>
        `;
        const input = form.querySelector('input');
        const button = form.querySelector('button');
        const status = form.querySelector('.infographic-email-status');
        input.value = prefill;
        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            const email = input.value.trim();
            if (!email) return;
            button.disabled = true;
            status.textContent = 'Sending...';
            status.className = 'form-text infographic-email-status text-muted';
            try {
                const body = { email, task_id: taskId || this.lastInfographicTaskId || null };
                if (opts.includeImage && this.infographicB64) body.image_b64 = this.infographicB64;
                const response = await fetch('/email_infographic', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(body)
                });
                const data = await this.parseJSONResponse(response);
                if (response.ok && data.status === 'scheduled') {
                    status.textContent = `We will email the PDF to ${data.email} as soon as it is ready.`;
                    status.className = 'form-text infographic-email-status text-success';
                    if (window.INFOGRAPHIC_EMAIL) window.INFOGRAPHIC_EMAIL.prefill = data.email;
                    this.pollInfographicEmailStatus(body.task_id, status, button);
                } else if (response.ok && data.status === 'sent') {
                    status.textContent = `Sent via ${data.transport || 'email'}! Check ${data.email} for the PDF (and your junk folder).`;
                    status.className = 'form-text infographic-email-status text-success';
                    if (window.INFOGRAPHIC_EMAIL) window.INFOGRAPHIC_EMAIL.prefill = data.email;
                } else {
                    status.textContent = data.error || 'Could not send the email. Please try again.';
                    status.className = 'form-text infographic-email-status text-danger';
                    button.disabled = false;
                }
            } catch (err) {
                console.error('Infographic email error:', err);
                status.textContent = 'Network error - please try again.';
                status.className = 'form-text infographic-email-status text-danger';
                button.disabled = false;
            }
        });
        container.appendChild(form);
    }

    // After a scheduled "email me when ready" request, poll for the delivery
    // outcome so a failure (e.g. the provider rejecting the address) is shown
    // instead of silently logged on the server.
    pollInfographicEmailStatus(taskId, statusEl, buttonEl) {
        if (!taskId) return;
        let attempts = 0;
        const maxAttempts = 180; // 15 minutes at 5s
        const poll = async () => {
            attempts++;
            try {
                const r = await fetch(`/infographic_email_status/${taskId}`);
                const d = await r.json();
                if (d.status === 'sent') {
                    statusEl.textContent = `Sent! Check ${d.email} for the PDF (and your junk folder).`;
                    statusEl.className = 'form-text infographic-email-status text-success';
                    return;
                }
                if (d.status === 'failed') {
                    statusEl.textContent = d.error || 'Sending the email failed.';
                    statusEl.className = 'form-text infographic-email-status text-danger';
                    if (buttonEl) buttonEl.disabled = false;
                    return;
                }
            } catch (e) {
                console.warn('Email status poll error:', e);
            }
            if (attempts < maxAttempts) setTimeout(poll, 5000);
        };
        setTimeout(poll, 5000);
    }

    downloadInfographic() {
        if (!this.infographicB64) return;
        // Decode base64 to a Blob: object URLs handle multi-megabyte files far
        // more reliably than data: URLs on anchor downloads.
        const byteString = atob(this.infographicB64);
        const bytes = new Uint8Array(byteString.length);
        for (let i = 0; i < byteString.length; i++) {
            bytes[i] = byteString.charCodeAt(i);
        }
        const blob = new Blob([bytes], { type: 'image/png' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'revision-infographic.png';
        document.body.appendChild(a);
        a.click();
        a.remove();
        setTimeout(() => URL.revokeObjectURL(url), 10000);
    }

    openInfographicViewer() {
        if (!this.infographicB64) return;

        const overlay = document.createElement('div');
        overlay.className = 'infographic-viewer';
        overlay.innerHTML = `
            <div class="infographic-toolbar">
                <button type="button" data-act="zoomout" title="Zoom out" aria-label="Zoom out"><i class="fas fa-search-minus"></i></button>
                <span class="infographic-zoom-label">100%</span>
                <button type="button" data-act="zoomin" title="Zoom in" aria-label="Zoom in"><i class="fas fa-search-plus"></i></button>
                <button type="button" data-act="fit" title="Fit to screen">Fit</button>
                <button type="button" data-act="actual" title="Actual size">1:1</button>
                <button type="button" data-act="download" title="Download PNG" aria-label="Download PNG"><i class="fas fa-download"></i></button>
                <button type="button" data-act="close" title="Close (Esc)" aria-label="Close viewer"><i class="fas fa-times"></i></button>
            </div>
            <div class="infographic-stage">
                <img alt="Revision guide infographic" draggable="false">
            </div>
        `;
        document.body.appendChild(overlay);
        document.body.style.overflow = 'hidden';

        const stage = overlay.querySelector('.infographic-stage');
        const img = overlay.querySelector('img');
        const zoomLabel = overlay.querySelector('.infographic-zoom-label');

        let scale = 1, tx = 0, ty = 0, fitScale = 1;
        const MAX_SCALE = 8;

        const apply = () => {
            img.style.transform = `translate(${tx}px, ${ty}px) scale(${scale})`;
            zoomLabel.textContent = Math.round(scale * 100) + '%';
        };

        const fit = () => {
            if (!img.naturalWidth) return;
            fitScale = Math.min(
                stage.clientWidth / img.naturalWidth,
                stage.clientHeight / img.naturalHeight
            ) * 0.98;
            scale = fitScale;
            tx = (stage.clientWidth - img.naturalWidth * scale) / 2;
            ty = (stage.clientHeight - img.naturalHeight * scale) / 2;
            apply();
        };

        // Zoom keeping the stage point (px, py) fixed under the cursor/fingers
        const zoomAt = (px, py, factor) => {
            const newScale = Math.min(MAX_SCALE, Math.max(fitScale * 0.5, scale * factor));
            factor = newScale / scale;
            tx = px - (px - tx) * factor;
            ty = py - (py - ty) * factor;
            scale = newScale;
            apply();
        };

        const zoomAtCenter = (factor) => zoomAt(stage.clientWidth / 2, stage.clientHeight / 2, factor);

        const close = () => {
            document.removeEventListener('keydown', onKeyDown);
            window.removeEventListener('resize', fit);
            document.body.style.overflow = '';
            overlay.remove();
        };

        const onKeyDown = (e) => {
            if (e.key === 'Escape') close();
            else if (e.key === '+' || e.key === '=') zoomAtCenter(1.25);
            else if (e.key === '-' || e.key === '_') zoomAtCenter(0.8);
        };
        document.addEventListener('keydown', onKeyDown);
        window.addEventListener('resize', fit);

        overlay.querySelector('[data-act="zoomin"]').addEventListener('click', () => zoomAtCenter(1.25));
        overlay.querySelector('[data-act="zoomout"]').addEventListener('click', () => zoomAtCenter(0.8));
        overlay.querySelector('[data-act="fit"]').addEventListener('click', fit);
        overlay.querySelector('[data-act="actual"]').addEventListener('click', () => zoomAtCenter(1 / scale));
        overlay.querySelector('[data-act="download"]').addEventListener('click', () => this.downloadInfographic());
        overlay.querySelector('[data-act="close"]').addEventListener('click', close);

        // Mouse-wheel / trackpad zoom centred on the cursor
        stage.addEventListener('wheel', (e) => {
            e.preventDefault();
            const rect = stage.getBoundingClientRect();
            zoomAt(e.clientX - rect.left, e.clientY - rect.top, Math.exp(-e.deltaY * 0.0015));
        }, { passive: false });

        // Drag-to-pan (one pointer) and pinch-to-zoom (two pointers)
        const pointers = new Map();
        let lastPinchDist = 0;

        stage.addEventListener('pointerdown', (e) => {
            e.preventDefault();
            stage.setPointerCapture(e.pointerId);
            pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
            if (pointers.size === 2) {
                const pts = [...pointers.values()];
                lastPinchDist = Math.hypot(pts[0].x - pts[1].x, pts[0].y - pts[1].y);
            }
            stage.classList.add('dragging');
        });

        stage.addEventListener('pointermove', (e) => {
            if (!pointers.has(e.pointerId)) return;
            const prev = pointers.get(e.pointerId);
            pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });

            if (pointers.size === 1) {
                tx += e.clientX - prev.x;
                ty += e.clientY - prev.y;
                apply();
            } else if (pointers.size === 2) {
                const pts = [...pointers.values()];
                const dist = Math.hypot(pts[0].x - pts[1].x, pts[0].y - pts[1].y);
                const rect = stage.getBoundingClientRect();
                const midX = (pts[0].x + pts[1].x) / 2 - rect.left;
                const midY = (pts[0].y + pts[1].y) / 2 - rect.top;
                if (lastPinchDist > 0) {
                    zoomAt(midX, midY, dist / lastPinchDist);
                }
                lastPinchDist = dist;
            }
        });

        const endPointer = (e) => {
            pointers.delete(e.pointerId);
            lastPinchDist = 0;
            if (pointers.size === 0) stage.classList.remove('dragging');
        };
        stage.addEventListener('pointerup', endPointer);
        stage.addEventListener('pointercancel', endPointer);

        // Double-click / double-tap toggles between fit and a readable zoom
        stage.addEventListener('dblclick', (e) => {
            const rect = stage.getBoundingClientRect();
            if (scale > fitScale * 1.4) {
                fit();
            } else {
                zoomAt(e.clientX - rect.left, e.clientY - rect.top, (fitScale * 2.5) / scale);
            }
        });

        img.addEventListener('load', fit);
        img.src = 'data:image/png;base64,' + this.infographicB64;
        if (img.complete) fit();
    }

    addMessage(role, content) {
        // Use the global addMessage function from templates
        if (typeof window.addMessage === 'function') {
            window.addMessage(role, content);
        } else {
            // Fallback if global function not available
            console.log(`${role}: ${content}`);
        }
    }

}

// Global functions for onclick handlers
function endCalculationSession() {
    if (window.aiTutor) {
        window.aiTutor.endCalculationPractice();
    }
}

function nextCalculationQuestion() {
    if (window.aiTutor) {
        window.aiTutor.nextCalculationQuestion();
    }
}

function endEssaySession() {
    if (window.aiTutor) window.aiTutor.endEssayPractice();
}

function nextEssayQuestion() {
    if (window.aiTutor) window.aiTutor.nextEssayQuestion();
}

async function submitEssayAnswer() {
    const textarea = document.getElementById('essayAnswerText');
    if (!textarea) return;
    const answer = textarea.value.trim();
    if (!answer) {
        alert('Please write your answer before submitting.');
        return;
    }
    if (answer.length > 5000) {
        alert('Your answer is too long to submit (maximum 5000 characters, roughly 800 words). Please shorten it.');
        return;
    }
    const submitBtn = document.getElementById('essaySubmitBtn');
    if (submitBtn) {
        submitBtn.disabled = true;
        submitBtn.setAttribute('aria-label', 'Marking your answer');
        submitBtn.innerHTML = '<div class="spinner-border spinner-border-sm" role="status"></div> Marking...';
    }
    textarea.disabled = true;

    // Send through the chat system; the server marks it because essay practice is active
    const chatInput = document.getElementById('newChatInput');
    if (chatInput) {
        chatInput.value = answer;
        await newSendMessage();
    } else {
        console.error('newChatInput not found; cannot submit essay answer');
    }

    textarea.disabled = false;
    textarea.value = '';
    const counter = document.getElementById('essayWordCount');
    if (counter) counter.textContent = '0 words';
    if (submitBtn) {
        submitBtn.disabled = false;
        submitBtn.removeAttribute('aria-label');
        submitBtn.innerHTML = '<i class="fas fa-check"></i> Check Answer';
    }
}

async function submitCalculationAnswer() {
    console.log('📝 GLOBAL DEBUG: submitCalculationAnswer called');
    
    const calcInput = document.getElementById('calcAnswerInput');
    if (!calcInput) {
        console.log('📝 GLOBAL DEBUG: calcAnswerInput not found');
        return;
    }
    
    const answer = calcInput.value.trim();
    if (!answer) {
        alert('Please enter an answer before submitting.');
        return;
    }

    // Show spinner inside the calculation section and disable button.
    // The spinner swap removes the button's visible text, so keep an
    // aria-label while it is showing.
    const loadingDiv = document.getElementById('calcAnswerLoading');
    const submitBtn = document.getElementById('calcSubmitBtn');
    if (loadingDiv) loadingDiv.style.display = 'block';
    if (submitBtn) {
        submitBtn.disabled = true;
        submitBtn.setAttribute('aria-label', 'Checking your answer');
        submitBtn.innerHTML = '<div class="spinner-border spinner-border-sm" role="status"></div>';
    }

    // Use the new chat system to send the answer
    const chatInput = document.getElementById('newChatInput');
    if (chatInput) {
        chatInput.value = answer;
        calcInput.value = '';
        await newSendMessage();
    } else {
        console.error('newChatInput not found; cannot submit calculation answer');
    }

    // Hide spinner and restore button
    if (loadingDiv) loadingDiv.style.display = 'none';
    if (submitBtn) {
        submitBtn.disabled = false;
        submitBtn.removeAttribute('aria-label');
        submitBtn.innerHTML = '<i class="fas fa-check"></i> Check Answer';
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    window.aiTutor = new AITutor();
});
